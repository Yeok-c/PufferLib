## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
import sys
import glob
import ast
import csv
import json
import time
import random
import traceback
import shutil
import socket
import subprocess
import platform
import argparse
import importlib
import configparser
from numbers import Number
from threading import Thread
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
import pufferlib.spaces
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME
)
# Assume advantage kernel has been built if torch has been compiled with CUDA or HIP support
# and can find CUDA or HIP in the system
ADVANTAGE_CUDA = bool(CUDA_HOME or ROCM_HOME)

@dataclass
class TrainerLogSnapshot:
    """Typed trainer metrics + dynamic env metrics."""
    sps: float
    agent_steps: int
    uptime: float
    epoch: int
    learning_rate: float
    environment: dict[str, Any]
    losses: dict[str, Any]
    performance: dict[str, Any]
    system: dict[str, Any]

    def to_logs(self):
        logs = {
            'SPS': self.sps,
            'agent_steps': self.agent_steps,
            'uptime': self.uptime,
            'epoch': self.epoch,
            'learning_rate': self.learning_rate,
        }
        logs.update({f'environment/{k}': v for k, v in self.environment.items()})
        logs.update({f'losses/{k}': v for k, v in self.losses.items()})
        logs.update({f'performance/{k}': v for k, v in self.performance.items()})
        logs.update({f'system/{k}': v for k, v in self.system.items()})
        return logs

class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.deterministic = config['torch_deterministic']
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config['seed']
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.manual_seed(seed)

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        # Experience
        if config['batch_size'] == 'auto' and config['bptt_horizon'] == 'auto':
            raise pufferlib.APIUsageError('Must specify batch_size or bptt_horizon')
        elif config['batch_size'] == 'auto':
            config['batch_size'] = total_agents * config['bptt_horizon']
        elif config['bptt_horizon'] == 'auto':
            config['bptt_horizon'] = config['batch_size'] // total_agents

        batch_size = config['batch_size']
        horizon = config['bptt_horizon']
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f'Total agents {total_agents} <= segments {segments}'
            )

        device = config['device']
        self.observations = torch.zeros(segments, horizon, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and config['cpu_offload'],
            device='cpu' if config['cpu_offload'] else device)
        self.actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents

        # LSTM
        if config['use_rnn']:
            n = vecenv.agents_per_batch
            h = policy.hidden_size
            self.lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
            self.lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}

        # Minibatching & gradient accumulation
        minibatch_size = config['minibatch_size']
        max_minibatch_size = config['max_minibatch_size']
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise pufferlib.APIUsageError(
                f'minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly')

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(
                f'batch_size {batch_size} must be >= minibatch_size {minibatch_size}'
            )

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(config['update_epochs'] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon 
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f'minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}'
            )

        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy
        if config['compile']:
            self.policy = torch.compile(policy, mode=config['compile_mode'])
            self.policy.forward_eval = torch.compile(policy, mode=config['compile_mode'])
            pufferlib.pytorch.sample_logits = torch.compile(pufferlib.pytorch.sample_logits, mode=config['compile_mode'])

        # Optimizer
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
            )
        elif config['optimizer'] == 'muon':
            import heavyball
            from heavyball import ForeachMuon
            warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')
            heavyball.utils.compile_mode = "default"

            # # optionally a little bit better/faster alternative to newtonschulz iteration
            # import heavyball.utils
            # heavyball.utils.zeroth_power_mode = 'thinky_polar_express'

            # heavyball_momentum=True introduced in heavyball 2.1.1
            # recovers heavyball-1.7.2 behaviour - previously swept hyperparameters work well
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
                heavyball_momentum=True,
            )
        else:
            raise ValueError(f'Unknown optimizer: {config["optimizer"]}')

        self.optimizer = optimizer

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        # Learning rate scheduler
        epochs = config['total_timesteps'] // config['batch_size']
        eta_min = config['learning_rate'] * config['min_lr_ratio']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min)
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config['precision']
        self.amp_context = contextlib.nullcontext()
        if config.get('amp', True) and config['device'] == 'cuda':
            self.amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, precision))
        if precision not in ('float32', 'bfloat16'):
            raise pufferlib.APIUsageError(f'Invalid precision: {precision}: use float32 or bfloat16')

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def _sample_random_actions(self, batch_size, device):
        atn_space = self.vecenv.single_action_space
        if isinstance(atn_space, pufferlib.spaces.Discrete):
            action = torch.randint(
                low=0,
                high=atn_space.n,
                size=(batch_size, *atn_space.shape),
                device=device,
                dtype=self.actions.dtype,
            )
        elif isinstance(atn_space, pufferlib.spaces.MultiDiscrete):
            nvec = torch.as_tensor(atn_space.nvec, device=device)
            action = torch.floor(torch.rand(batch_size, len(atn_space.nvec), device=device) * nvec).to(self.actions.dtype)
        elif isinstance(atn_space, pufferlib.spaces.Box):
            low = torch.as_tensor(atn_space.low, device=device, dtype=torch.float32)
            high = torch.as_tensor(atn_space.high, device=device, dtype=torch.float32)
            action = low + (high - low) * torch.rand((batch_size, *atn_space.shape), device=device)
            action = action.to(self.actions.dtype)
        else:
            raise pufferlib.APIUsageError(f'Unsupported action space for NoPolicy: {atn_space}')

        logprob = torch.zeros(batch_size, device=device)
        value = torch.zeros(batch_size, device=device)
        return action, logprob, value

    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        device = config['device']

        if config['use_rnn']:
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            done_mask = d + t # TODO: Handle truncations separately
            self.global_step += int(mask.sum())

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)#, non_blocking=True)
            r = torch.as_tensor(r).to(device)#, non_blocking=True)
            d = torch.as_tensor(d).to(device)#, non_blocking=True)

            profile('eval_forward', epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config['use_rnn']:
                    state['lstm_h'] = self.lstm_h[env_id.start]
                    state['lstm_c'] = self.lstm_c[env_id.start]

                if getattr(self.policy, 'is_no_policy', False):
                    action, logprob, value = self._sample_random_actions(o_device.shape[0], device)
                else:
                    logits, value = self.policy.forward_eval(o_device, state)
                    action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                r = torch.clamp(r, -1, 1)

            profile('eval_copy', epoch)
            with torch.no_grad():
                if config['use_rnn']:
                    self.lstm_h[env_id.start] = state['lstm_h']
                    self.lstm_c[env_id.start] = state['lstm_c']

                # Fast path for fully vectorized envs. Some vector backends report
                # env indices while returning batched agent rows (agents_per_batch > 1).
                # Derive the effective agent-row span from the returned tensor shape.
                env_count = max(1, env_id.stop - env_id.start)
                batch_count = int(o_device.shape[0])
                agents_per_batch = getattr(self.vecenv, 'agents_per_batch', None)
                if agents_per_batch is None or agents_per_batch * env_count != batch_count:
                    agents_per_batch = max(1, batch_count // env_count)

                agent_start = env_id.start * agents_per_batch
                agent_stop = agent_start + batch_count
                agent_id = slice(agent_start, agent_stop)

                l = self.ep_lengths[agent_id.start].item()
                batch_rows = slice(self.ep_indices[agent_id.start].item(), 1+self.ep_indices[agent_id.stop - 1].item())

                if config['cpu_offload']:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l+1 >= config['bptt_horizon']:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config['device']).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(self.vecenv.single_action_space, pufferlib.spaces.Box):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile('eval_misc', epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile('env', epoch)
            self.vecenv.send(action)

        profile('eval_misc', epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        profile('train_misc', epoch, nest=True)
        losses = defaultdict(float)
        config = self.config
        device = config['device']

        if getattr(self.policy, 'is_no_policy', False):
            self.losses = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'old_approx_kl': 0.0,
                'approx_kl': 0.0,
                'clipfrac': 0.0,
                'importance': 1.0,
                'explained_variance': 0.0,
            }
            profile.end()
            logs = None
            self.epoch += 1
            done_training = self.global_step >= config['total_timesteps']
            if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
                logs = self.mean_and_log()
                self.print_dashboard()
                self.stats = defaultdict(list)
                self.last_log_time = time.time()
                self.last_log_step = self.global_step
                profile.clear()

            if self.epoch % config['checkpoint_interval'] == 0 or done_training:
                self.save_checkpoint()
                self.msg = f'Checkpoint saved at update {self.epoch}'

            return logs

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile('train_misc', epoch)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(self.values, self.rewards,
                self.terminals, self.ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Prioritize experience by advantage magnitude
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta

            profile('train_copy', epoch)
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile('train_forward', epoch)
            if not config['use_rnn']:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            logits, newvalue = self.policy(mb_obs, state)
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

            profile('train_misc', epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            # NOTE: Commenting this out since adv is replaced below
            # adv = advantages[idx]
            # adv = compute_puff_advantage(mb_values, mb_rewards, mb_terminals,
            #     ratio, adv, config['gamma'], config['gae_lambda'],
            #     config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Weight advantages by priority and normalize
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss
            self.amp_context.__enter__() # TODO: AMP needs some debugging

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()

            # Logging
            profile('train_misc', epoch)
            losses['policy_loss'] += pg_loss.item() / self.total_minibatches
            losses['value_loss'] += v_loss.item() / self.total_minibatches
            losses['entropy'] += entropy_loss.item() / self.total_minibatches
            losses['old_approx_kl'] += old_approx_kl.item() / self.total_minibatches
            losses['approx_kl'] += approx_kl.item() / self.total_minibatches
            losses['clipfrac'] += clipfrac.item() / self.total_minibatches
            losses['importance'] += ratio.mean().item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile('learn', epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Reprioritize experience
        profile('train_misc', epoch)
        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs

    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config['device']
        agent_steps = int(dist_sum(self.global_step, device))
        snapshot = TrainerLogSnapshot(
            sps=dist_sum(self.sps, device),
            agent_steps=agent_steps,
            uptime=time.time() - self.start_time,
            epoch=int(dist_sum(self.epoch, device)),
            learning_rate=self.optimizer.param_groups[0]["lr"],
            environment=dict(self.stats),
            losses=dict(self.losses),
            performance={k: v['elapsed'] for k, v in self.profile},
            system=self.utilization.snapshot_metrics(),
            # environment={k: dist_mean(v, device) for k, v in self.stats.items()},
            # losses={k: dist_mean(v, device) for k, v in self.losses.items()},
            # performance={k: dist_sum(v['elapsed'], device) for k, v in self.profile},
        )
        logs = snapshot.to_logs()

        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               self.logger.log(logs, agent_steps)
               return logs
           else:
               return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        log_dir = self.config.get('log_dir', self.config.get('data_dir', 'experiments'))
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f'{self.config["env"]}_{run_id}.pt')
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        run_id = self.logger.run_id
        log_dir = self.config.get('log_dir', self.config.get('data_dir', 'experiments'))
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        path = os.path.join(checkpoint_dir, f'{self.config["env"]}_{run_id}')
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)
        return model_path

    def print_dashboard(self, clear=False, idx=[0],
            c1='[cyan]', c2='[dim default]', b1='[bright_cyan]', b2='[default]'):
        config = self.config
        sps = dist_sum(self.sps, config['device'])
        agent_steps = dist_sum(self.global_step, config['device'])
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True,
            show_header=False, border_style='bright_cyan')
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f'{b1}PufferLib {b2}3.0 {idx[0]*" "}:blowfish:',
            f'{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%',
            f'{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%',
            f'{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%',
            f'{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%',
        )
        idx[0] = (idx[0] - 1) % 10
            
        s = Table(box=None, expand=True)
        remaining = f'{b2}A hair past a freckle{c2}'
        if sps != 0:
            remaining = duration((config['total_timesteps'] - agent_steps)/sps, b2, c2)

        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{b2}Env', f'{b2}{config["env"]}')
        s.add_row(f'{b2}Params', abbreviate(self.model_size, b2, c2))
        s.add_row(f'{b2}Steps', abbreviate(agent_steps, b2, c2))
        s.add_row(f'{b2}SPS', abbreviate(sps, b2, c2))
        s.add_row(f'{b2}Epoch', f'{b2}{self.epoch}')
        s.add_row(f'{b2}Uptime', duration(self.uptime, b2, c2))
        s.add_row(f'{b2}Remaining', remaining)

        delta = profile.eval['buffer'] + profile.train['buffer']
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf('  Env', b2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf('Train', b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf('  Learn', b2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.train_misc, b2, c2))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    '''CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version.'''

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()

    torch.ops.pufferlib.compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return f'{b2}{num}{c2}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}K'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}M'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}B'
    else:
        return f'{b2}{num/1e12:.2f}{c2}T'

def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()

def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()

class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        # Skip profiling the first few epochs, which are noisy due to setup
        if (epoch + 1) % self.frequency != 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]['start'] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile['start']
        profile['delta'] += delta
        # Multiply delta by freq to account for skipped epochs
        profile['elapsed'] += delta * self.frequency

    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof['delta'] > 0:
                prof['buffer'] = prof['delta']
                prof['delta'] = 0

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.cpu_core_util = deque([[]], maxlen=maxlen)
        self.cpu_power_watts = deque([float('nan')], maxlen=maxlen)
        self.ram_used_bytes = deque([0], maxlen=maxlen)
        self.ram_total_bytes = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.gpu_power_watts = deque([float('nan')], maxlen=maxlen)
        self.vram_used_bytes = deque([0], maxlen=maxlen)
        self.vram_total_bytes = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self._rapl_paths = glob.glob('/sys/class/powercap/intel-rapl:*/energy_uj')
        self._last_cpu_energy_uj = None
        self._last_cpu_energy_ts = None
        self._nvidia_smi_available = None
        self.start()

    def _mean_valid(self, values, default=float('nan')):
        vals = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return default
        return float(np.mean(vals))

    def _read_cpu_power_watts(self):
        if len(self._rapl_paths) == 0:
            return float('nan')

        try:
            energy_uj = 0
            for path in self._rapl_paths:
                with open(path) as f:
                    energy_uj += int(f.read().strip())

            now = time.time()
            if self._last_cpu_energy_uj is None or self._last_cpu_energy_ts is None:
                self._last_cpu_energy_uj = energy_uj
                self._last_cpu_energy_ts = now
                return float('nan')

            delta_uj = energy_uj - self._last_cpu_energy_uj
            delta_t = now - self._last_cpu_energy_ts
            self._last_cpu_energy_uj = energy_uj
            self._last_cpu_energy_ts = now
            if delta_uj <= 0 or delta_t <= 0:
                return float('nan')
            return (delta_uj / 1e6) / delta_t
        except Exception:
            return float('nan')

    def _read_gpu_power_watts(self):
        if self._nvidia_smi_available is False:
            return float('nan')

        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip().splitlines()
            self._nvidia_smi_available = True
            if len(out) == 0:
                return float('nan')
            return float(out[0].split(',')[0].strip())
        except Exception:
            self._nvidia_smi_available = False
            return float('nan')

    def run(self):
        while not self.stopped:
            per_core = psutil.cpu_percent(percpu=True)
            self.cpu_core_util.append(per_core)
            if len(per_core) > 0:
                self.cpu_util.append(float(np.mean(per_core)))
            else:
                self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())

            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            self.ram_used_bytes.append(mem.active)
            self.ram_total_bytes.append(mem.total)
            self.cpu_power_watts.append(self._read_cpu_power_watts())
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                   self.gpu_util.append(float('nan'))
                   self.gpu_mem.append(float('nan'))
                   self.gpu_power_watts.append(float('nan'))
                   self.vram_used_bytes.append(0)
                   self.vram_total_bytes.append(0)
                   time.sleep(self.delay)
                   continue

                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100*(total-free)/total)
                self.vram_used_bytes.append(total - free)
                self.vram_total_bytes.append(total)
                self.gpu_power_watts.append(self._read_gpu_power_watts())
            else:
                self.gpu_util.append(float('nan'))
                self.gpu_mem.append(float('nan'))
                self.gpu_power_watts.append(float('nan'))
                self.vram_used_bytes.append(0)
                self.vram_total_bytes.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

    def snapshot_metrics(self):
        per_core = self.cpu_core_util[-1] if len(self.cpu_core_util) > 0 else []
        core_metrics = {f'cpu_core_{i}_pct': value for i, value in enumerate(per_core)}
        metrics = {
            'cpu_pct': self._mean_valid(self.cpu_util, 0.0),
            'ram_pct': self._mean_valid(self.cpu_mem, 0.0),
            'gpu_pct': self._mean_valid(self.gpu_util),
            'vram_pct': self._mean_valid(self.gpu_mem),
            'cpu_power_watts': self._mean_valid(self.cpu_power_watts),
            'gpu_power_watts': self._mean_valid(self.gpu_power_watts),
            'ram_used_bytes': int(self.ram_used_bytes[-1]),
            'ram_total_bytes': int(self.ram_total_bytes[-1]),
            'vram_used_bytes': int(self.vram_used_bytes[-1]),
            'vram_total_bytes': int(self.vram_total_bytes[-1]),
            **core_metrics,
        }
        return metrics

def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]

class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100*time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path, early_stop):
        pass

class CsvLogger:
    def __init__(self, args, env_name=None, policy=None, vecenv=None):
        self.run_id = str(int(100*time.time()))
        self.env_name = env_name
        self.args = args
        self.policy = policy
        self.vecenv = vecenv
        csv_dir = args.get('log_dir')
        if csv_dir is None:
            raise pufferlib.APIUsageError('log_dir is required when using CsvLogger')

        if os.path.exists(csv_dir) and not os.path.isdir(csv_dir):
            raise pufferlib.APIUsageError(
                f'log_dir must point to a directory, but found a file: {csv_dir}'
            )

        os.makedirs(csv_dir, exist_ok=True)

        self.path = os.path.join(csv_dir, 'logs.csv')
        self.schema_path = os.path.join(csv_dir, 'schema.csv')
        self.columns = []
        self.field_types = {}
        self.file = open(self.path, 'a+', newline='')
        self.file.seek(0)
        self._load_existing_header()
        self._write_schema()

        self._write_experiment_info_csv(csv_dir)

    def _machine_info(self):
        policy_name = self._policy_name()
        grid_size = self._grid_size()
        cpu_freq = float('nan')
        try:
            psutil_freq = psutil.cpu_freq()
            if psutil_freq is not None and psutil_freq.current is not None:
                cpu_freq = float(psutil_freq.current) * 1_000_000.0
        except Exception:
            pass

        # CPU package power is not exposed consistently across platforms via psutil.
        cpu_total_power_watts = float('nan')
        cpu_model_name = self._cpu_model_name()
        cpu_governor_or_power_mode = self._cpu_governor_or_power_mode()

        gpu_name = ''
        gpu_total_watts = float('nan')
        gpu_driver_version = ''
        gpu_clock_graphics_mhz = float('nan')
        gpu_clock_mem_mhz = float('nan')
        nvidia_fields = self._nvidia_smi_fields()
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_total_watts = nvidia_fields.get('power.limit', float('nan'))
            gpu_driver_version = nvidia_fields.get('driver_version', '')
            gpu_clock_graphics_mhz = nvidia_fields.get('clocks.gr', float('nan'))
            gpu_clock_mem_mhz = nvidia_fields.get('clocks.mem', float('nan'))
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_name = 'Apple Metal (MPS)'

        cuda_runtime_version = torch.version.cuda if torch.version.cuda is not None else ''
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version is None:
            cudnn_version = ''

        info = {
            'run_id': self.run_id,
            'env': self.env_name if self.env_name is not None else '',
            'policy_name': policy_name,
            'grid_size': grid_size,
            'timestamp': time.time(),
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'git_commit': self._git_commit(),
            'torch_version': torch.__version__,
            'torch_build_cuda': torch.version.cuda if torch.version.cuda is not None else '',
            'cuda_runtime_version': cuda_runtime_version,
            'cudnn_version': cudnn_version,
            'pufferlib_version': getattr(pufferlib, '__version__', ''),
            'python_version': platform.python_version(),
            'cpu_model_name': cpu_model_name,
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_freq': cpu_freq,
            'cpu_governor_or_power_mode': cpu_governor_or_power_mode,
            'cpu_total_power_watts': cpu_total_power_watts,
            'memory_total_bytes': psutil.virtual_memory().total,
            'ram_speed': float('nan'),
            'swap_enabled': psutil.swap_memory().total > 0,
            'thermal_state': self._thermal_state(),
            'is_thermal_throttled': self._is_thermal_throttled(),
            'gpu_name': gpu_name,
            'gpu_total_watts': gpu_total_watts,
            'gpu_power_limit_watts': gpu_total_watts,
            'gpu_driver_version': gpu_driver_version,
            'gpu_clock_graphics_mhz': gpu_clock_graphics_mhz,
            'gpu_clock_mem_mhz': gpu_clock_mem_mhz,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            info['cuda_device_name_0'] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            info['cuda_vram_total_bytes_0'] = total
            info['cuda_vram_free_bytes_0'] = free
            info['cuda_power_limit_watts_0'] = self._gpu_power_limit_watts()
        return info

    def _policy_name(self):
        if hasattr(self.args, 'get'):
            value = self.args.get('policy_name')
            if value:
                return str(value)
        if self.policy is not None:
            return self.policy.__class__.__name__
        return ''

    def _grid_size(self):
        if not hasattr(self.args, 'get'):
            return ''

        env_args = self.args.get('env')
        if hasattr(env_args, 'get'):
            value = env_args.get('grid_size')
            if value is not None:
                return value

        for key in ('env.grid_size', 'env.grid-size', 'grid_size', 'grid-size'):
            value = self.args.get(key)
            if value is not None:
                return value

        return ''

    def _model_info(self):
        if self.policy is None:
            return {
                'model_param_count_total': float('nan'),
                'model_param_count_trainable': float('nan'),
                'model_param_size_bytes': float('nan'),
                'model_param_size_mib': float('nan'),
            }

        total_params = 0
        trainable_params = 0
        param_size_bytes = 0
        for param in self.policy.parameters():
            count = int(param.numel())
            total_params += count
            if param.requires_grad:
                trainable_params += count
            param_size_bytes += count * int(param.element_size())

        return {
            'model_param_count_total': total_params,
            'model_param_count_trainable': trainable_params,
            'model_param_size_bytes': param_size_bytes,
            'model_param_size_mib': float(param_size_bytes / (1024 * 1024)),
        }

    def _experiment_info(self):
        info = self._machine_info()
        info.update(self._model_info())
        info.update(self._spaces_and_arch_info())
        return info

    def _spaces_and_arch_info(self):
        obs_space = None
        action_space = None
        if self.vecenv is not None:
            obs_space = getattr(self.vecenv, 'single_observation_space', None)
            action_space = getattr(self.vecenv, 'single_action_space', None)

        return {
            'observation_space': self._space_to_string(obs_space),
            'action_space': self._space_to_string(action_space),
            'neural_network_final_layer_size': self._infer_final_layer_size(),
        }

    def _space_to_string(self, space):
        if space is None:
            return ''
        return str(space)

    def _infer_final_layer_size(self):
        if self.policy is None:
            return float('nan')

        # Use the last trainable layer with explicit output size if possible.
        for module in reversed(list(self.policy.modules())):
            out_features = getattr(module, 'out_features', None)
            if out_features is not None:
                return int(out_features)

            out_channels = getattr(module, 'out_channels', None)
            if out_channels is not None:
                return int(out_channels)

        return float('nan')

    def _gpu_power_limit_watts(self):
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip().splitlines()
            if len(out) == 0:
                return float('nan')
            return float(out[0].split(',')[0].strip())
        except Exception:
            return float('nan')

    def _nvidia_smi_fields(self):
        keys = ['driver_version', 'power.limit', 'clocks.gr', 'clocks.mem']
        result = {
            'driver_version': '',
            'power.limit': float('nan'),
            'clocks.gr': float('nan'),
            'clocks.mem': float('nan'),
        }
        try:
            out = subprocess.check_output(
                ['nvidia-smi', f'--query-gpu={",".join(keys)}', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip().splitlines()
            if len(out) == 0:
                return result

            values = [v.strip() for v in out[0].split(',')]
            if len(values) != len(keys):
                return result

            result['driver_version'] = values[0]
            result['power.limit'] = float(values[1])
            result['clocks.gr'] = float(values[2])
            result['clocks.mem'] = float(values[3])
            return result
        except Exception:
            return result

    def _git_commit(self):
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return ''

    def _cpu_model_name(self):
        try:
            if platform.system() == 'Darwin':
                return subprocess.check_output(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo') as cpuinfo:
                    for line in cpuinfo:
                        if 'model name' in line:
                            return line.split(':', 1)[1].strip()
        except Exception:
            pass
        return platform.processor()

    def _cpu_governor_or_power_mode(self):
        try:
            if platform.system() == 'Linux':
                path = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
                if os.path.exists(path):
                    with open(path) as f:
                        return f.read().strip()
            if platform.system() == 'Darwin':
                out = subprocess.check_output(
                    ['pmset', '-g', 'custom'],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if out:
                    return out
        except Exception:
            pass
        return ''

    def _thermal_state(self):
        try:
            if platform.system() == 'Darwin':
                out = subprocess.check_output(
                    ['pmset', '-g', 'therm'],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if out:
                    return out
        except Exception:
            pass
        return ''

    def _is_thermal_throttled(self):
        therm = self._thermal_state().lower()
        return 'throttle' in therm

    def _write_experiment_info_csv(self, csv_dir):
        info_path = os.path.join(csv_dir, 'experiment_info.csv')
        experiment_info = self._experiment_info()
        with open(info_path, 'w', newline='') as info_file:
            writer = csv.writer(info_file)
            writer.writerow(['key', 'value'])
            for key, value in experiment_info.items():
                writer.writerow([key, self._serialize_value(value)])

    def _load_existing_header(self):
        self.file.seek(0)
        header = self.file.readline().strip()
        if header:
            self.columns = next(csv.reader([header]))
            if self.columns == ['step', 'metric', 'value', 'wall_time']:
                raise pufferlib.APIUsageError(
                    f'{self.path} uses the old long CSV format. '
                    'Please use a new log_dir (or remove the file) for wide per-step rows.'
                )
            if 'step' not in self.columns or 'wall_time' not in self.columns:
                raise pufferlib.APIUsageError(
                    f'Unexpected CSV header in {self.path}. Expected "step" and "wall_time" columns.'
                )
        else:
            self.columns = ['run_id', 'step', 'wall_time']
            self.file.seek(0)
            writer = csv.writer(self.file)
            writer.writerow(self.columns)
            self.file.flush()
            self._write_schema()

    def _infer_type(self, value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                return 'json'
        if isinstance(value, bool):
            return 'bool'
        if isinstance(value, int):
            return 'int'
        if isinstance(value, float):
            return 'float'
        if isinstance(value, str):
            return 'str'
        return 'json'

    def _write_schema(self):
        existing_rows = {}
        if os.path.exists(self.schema_path):
            with open(self.schema_path, newline='') as schema_file:
                reader = csv.DictReader(schema_file)
                for row in reader:
                    existing_rows[row['field']] = row

        with open(self.schema_path, 'w', newline='') as schema_file:
            writer = csv.writer(schema_file)
            writer.writerow(['field', 'type', 'format', 'source'])
            for field in self.columns:
                row = existing_rows.get(field, {})
                dtype = row.get('type') or self.field_types.get(field, 'str')
                fmt = row.get('format') or ('{:.4e}' if dtype in ('float', 'int') else 'raw')
                source = row.get('source') or ('system' if field in ('run_id', 'step', 'wall_time') else 'trainer')
                writer.writerow([field, dtype, fmt, source])

    def _rewrite_csv_with_new_columns(self, new_columns):
        self.file.seek(0)
        rows = []
        reader = csv.DictReader(self.file)
        for row in reader:
            rows.append(row)

        self.file.close()
        self.file = open(self.path, 'w+', newline='')
        writer = csv.DictWriter(self.file, fieldnames=new_columns)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, '') for field in new_columns}
            writer.writerow(out)
        self.file.flush()
        self.columns = new_columns
        self._write_schema()

    def _ensure_columns(self, logs):
        missing = [k for k in logs if k not in self.columns]
        if not missing:
            return

        for field in missing:
            self.field_types[field] = self._infer_type(logs[field])

        new_columns = [*self.columns, *missing]
        self._rewrite_csv_with_new_columns(new_columns)

    def _format_numeric(self, value):
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, Number):
            return f'{float(value):.4e}'
        return value

    def _serialize_value(self, value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return json.dumps(value.detach().cpu().tolist())
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value)
        return value

    def log(self, logs, step):
        logs = dict(logs)
        logs['run_id'] = self.run_id
        self._ensure_columns(logs)
        row = {field: '' for field in self.columns}
        row['run_id'] = self.run_id
        row['step'] = str(step)
        row['wall_time'] = self._format_numeric(time.time())
        for field, value in logs.items():
            serialized = self._serialize_value(value)
            row[field] = self._format_numeric(serialized)

        writer = csv.DictWriter(self.file, fieldnames=self.columns)
        writer.writerow(row)
        self.file.flush()

    def close(self, model_path, early_stop):
        if not self.file.closed:
            self.file.close()

class CompositeLogger:
    def __init__(self, loggers):
        if len(loggers) == 0:
            raise pufferlib.APIUsageError('CompositeLogger requires at least one logger')

        self.loggers = loggers
        self.run_id = loggers[0].run_id

    def log(self, logs, step):
        for logger in self.loggers:
            logger.log(logs, step)

    def close(self, model_path, early_stop):
        for logger in self.loggers:
            logger.close(model_path, early_stop)

class TensorboardLogger:
    def __init__(self, args):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            raise pufferlib.APIUsageError(
                'TensorBoard logging requires tensorboard. Install with `pip install tensorboard`.'
            ) from exc

        self.run_id = str(int(100*time.time()))
        log_dir = args.get('log_dir')
        if log_dir is None:
            raise pufferlib.APIUsageError('log_dir is required when using TensorboardLogger')
        base_dir = os.path.join(log_dir, 'tensorboard')
        env_name = args.get('env_name') or args['train'].get('env') or 'run'
        run_dir = os.path.join(base_dir, f'{env_name}_{self.run_id}')
        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=run_dir)

    def log(self, logs, step):
        for key, value in logs.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool):
                self.writer.add_scalar(key, int(value), step)
            elif isinstance(value, Number):
                self.writer.add_scalar(key, float(value), step)

    def close(self, model_path, early_stop):
        self.writer.add_text('run/model_path', str(model_path))
        self.writer.add_text('run/early_stop', str(early_stop))
        self.writer.flush()
        self.writer.close()

class NeptuneLogger:
    def __init__(self, args, load_id=None, mode='async'):
        import neptune as nept
        neptune_name = args['neptune_name']
        neptune_project = args['neptune_project']
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path):
        self.neptune['model'].track_files(model_path)

    def close(self, model_path, early_stop):
        self.neptune['early_stop'] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination='artifacts')
        return f'artifacts/{self.run_id}.pt'
 
class WandbLogger:
    def __init__(self, args, load_id=None, resume='allow'):
        import wandb
        wandb.init(
            id=load_id or wandb.util.generate_id(),
            project=args['wandb_project'],
            group=args['wandb_group'],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            tags = [args['tag']] if args['tag'] is not None else [],
            settings=wandb.Settings(console="off"),  # stop sending dashboard to wandb
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type='model')
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path, early_stop):
        self.wandb.run.summary['early_stop'] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f'{data_dir}/{model_file}'

def train(env_name, args=None, vecenv=None, policy=None, logger=None, early_stop_fn=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print("World size", world_size)
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if 'LOCAL_RANK' in os.environ:
        args['train']['device'] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend='nccl', world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank
        )
        if hasattr(policy, 'lstm'):
            #model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if logger is None:
        selected_loggers = []
        if args.get('log_dir'):
            selected_loggers.append(
                CsvLogger(args, env_name=env_name, policy=policy, vecenv=vecenv)
            )
        if args['tensorboard']:
            selected_loggers.append(TensorboardLogger(args))
        if args['neptune']:
            selected_loggers.append(NeptuneLogger(args))
        elif args['wandb']:
            selected_loggers.append(WandbLogger(args))

        if len(selected_loggers) == 1:
            logger = selected_loggers[0]
        elif len(selected_loggers) > 1:
            logger = CompositeLogger(selected_loggers)

    train_config = { **args['train'], 'env': env_name, 'log_dir': args.get('log_dir') }
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    # Sweep needs data for early stopped runs, so send data when steps > 100M
    logging_threshold = min(0.20*train_config['total_timesteps'], 100_000_000)
    max_logging_steps = 10
    logging_steps = 0
    all_logs = []

    while pufferl.global_step < train_config['total_timesteps']:
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            logging_steps += 1
            should_stop_early = False
            if early_stop_fn is not None:
                should_stop_early = early_stop_fn(logs)
                # This is hacky, but need to see if threshold looks reasonable
                if 'early_stop_threshold' in logs:
                    pufferl.logger.log({'environment/early_stop_threshold': logs['early_stop_threshold']}, logs['agent_steps'])

            if pufferl.global_step > logging_threshold:
                all_logs.append(logs)

            if should_stop_early:
                model_path = pufferl.close()
                pufferl.logger.close(model_path, early_stop=True)
                return all_logs

            if logging_steps >= max_logging_steps:
                model_path = pufferl.close()
                pufferl.logger.close(model_path, early_stop=False)
                return all_logs

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    for i in range(128):  # Run eval for at least 32, but put a hard stop at 128.
        stats = pufferl.evaluate()
        if i >= 32 and stats:
            break

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path, early_stop=False)
    return all_logs

def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    backend = args['vec']['backend']
    if backend != 'PufferEnv':
        backend = 'Serial'

    args['vec'] = dict(backend=backend, num_envs=1)
    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args['train']['device']

    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    frames = []
    dropped_frames = 0
    while True:
        render = driver.render()
        if len(frames) < args['save_frames']:
            if render is not None:
                frames.append(render)
            else:
                dropped_frames += 1
                if dropped_frames == 1:
                    print(
                        'Warning: render() returned None while collecting GIF frames. '
                        'Use --render-mode rgb_array for GIF capture.'
                    )

        # Screenshot Ocean envs with F12, gifs with control + F12
        if driver.render_mode == 'ansi':
            print('\033[0;0H' + render + '\n')
            time.sleep(1/args['fps'])
        elif driver.render_mode == 'rgb_array':
            pass
            #import cv2
            #render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            #cv2.imshow('frame', render)
            #cv2.waitKey(1)
            #time.sleep(1/args['fps'])

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob = vecenv.step(action)[0]

        if args['save_frames'] > 0 and len(frames) == 0 and dropped_frames >= args['save_frames']:
            vecenv.close()
            raise pufferlib.APIUsageError(
                'Unable to save GIF: no renderable frames were produced. '
                'Try --render-mode rgb_array.'
            )

        if len(frames) > 0 and len(frames) == args['save_frames']:
            import imageio
            imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
            print(f'Saved {len(frames)} frames to {args["gif_path"]}')
            vecenv.close()
            return

def stop_if_loss_nan(logs):
    return any("losses/" in k and np.isnan(v) for k, v in logs.items())

def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args['wandb'] and not args['neptune'] and not args['tensorboard']:
        raise pufferlib.APIUsageError('Sweeps require at least one logger: wandb, neptune, or tensorboard')
    args['no_model_upload'] = True  # Uploading trained model during sweep crashed wandb

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    metric_name = args['sweep']['metric']
    # Support both prefixed metrics (e.g. "environment/score"), top-level trainer
    # metrics (e.g. "SPS"), and shorthand environment metrics (e.g. "score").
    top_level_metrics = {'SPS', 'agent_steps', 'uptime', 'epoch', 'learning_rate'}
    top_level_aliases = {'sps': 'SPS'}
    if '/' in metric_name:
        target_key = metric_name
    elif metric_name in top_level_metrics:
        target_key = metric_name
    elif metric_name.lower() in top_level_aliases:
        target_key = top_level_aliases[metric_name.lower()]
    else:
        target_key = f'environment/{metric_name}'
    running_target_buffer = deque(maxlen=30)

    def cleanup_failed_run_state():
        # If training crashes before logger.close(), ensure wandb does not keep
        # an active run alive and block the next sweep iteration in-process.
        if args['wandb']:
            with contextlib.suppress(Exception):
                import wandb
                if wandb.run is not None:
                    wandb.finish(exit_code=1, quiet=True)

        # OOM and driver-side failures can leave fragmented cache behind.
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    def stop_if_perf_below(logs):
        if stop_if_loss_nan(logs):
            logs['is_loss_nan'] = True
            return True

        if method != 'Protein':
            return False

        if ('uptime' in logs and target_key in logs):
            metric_val, cost = logs[target_key], logs['uptime']
            running_target_buffer.append(metric_val)
            target_running_mean = np.mean(running_target_buffer)
            
            # If metric distribution is percentile, threshold is also logit transformed
            threshold = sweep.get_early_stop_threshold(cost)
            logs['early_stop_threshold'] = max(threshold, -5)  # clipping for visualization

            if sweep.should_stop(max(target_running_mean, metric_val), cost):
                logs['is_loss_nan'] = False
                return True
        return False

    max_consecutive_failures = 5
    consecutive_failures = 0

    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # In the first run, skip sweep and use the train args specified in the config
        if i > 0:
            sweep.suggest(args)

        try:
            all_logs = train(env_name, args=args, early_stop_fn=stop_if_perf_below)
        except Exception:
            # Treat invalid hyperparameter combinations and runtime failures as sweep failures.
            # This keeps the sweep alive and allows the optimizer to avoid bad regions.
            cleanup_failed_run_state()
            warnings.warn(
                'Sweep run failed; continuing to next suggestion.\n'
                f'{traceback.format_exc()}',
                UserWarning,
                stacklevel=2,
            )
            sweep.observe(args, 0, 0, is_failure=True)
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                warnings.warn(
                    f'Stopping sweep after {consecutive_failures} consecutive failed runs.',
                    UserWarning,
                    stacklevel=2,
                )
                break
            continue

        all_logs = [e for e in all_logs if target_key in e]

        if not all_logs:
            sweep.observe(args, 0, 0, is_failure=True)
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                warnings.warn(
                    f'Stopping sweep after {consecutive_failures} consecutive failed runs.',
                    UserWarning,
                    stacklevel=2,
                )
                break
            continue

        total_timesteps = args['train']['total_timesteps']

        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log['uptime'] for log in all_logs], points_per_run)
        timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)

        is_final_loss_nan = all_logs[-1].get('is_loss_nan', False)
        if is_final_loss_nan:
            s = scores.pop()
            c = costs.pop()
            args['train']['total_timesteps'] = timesteps.pop()
            sweep.observe(args, s, c, is_failure=True)
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                warnings.warn(
                    f'Stopping sweep after {consecutive_failures} consecutive failed runs.',
                    UserWarning,
                    stacklevel=2,
                )
                break
        else:
            consecutive_failures = 0

        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps

def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args['train'], env=args['env_name'], tag=args['tag'])
    pufferl = PuffeRL(train_config, vecenv, policy, neptune=args['neptune'], wandb=args['wandb'])

    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    prof.export_chrome_trace("trace.json")

def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    args['vec'] = dict(backend='Serial', num_envs=1)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    path = f'{args["env_name"]}_weights.bin'
    weights = np.concatenate(weights)
    weights.tofile(path)
    print(f'Saved {len(weights)} weights to {path}')

def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    env_name = args['env_name']
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
 
def load_env(env_name, args):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    env_kwargs = dict(args['env'])
    render_mode = args.get('render_mode')
    if render_mode and render_mode != 'auto':
        env_kwargs['render_mode'] = None if render_mode == 'None' else render_mode
    return pufferlib.vector.make(make_env, env_kwargs=env_kwargs, **args['vec'])

def load_policy(args, vecenv, env_name=''):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)

    device = args['train']['device']
    policy_cls = getattr(env_module.torch, args['policy_name'])
    policy = policy_cls(vecenv.driver_env, **args['policy'])

    rnn_name = args['rnn_name']
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])
        policy = rnn_cls(vecenv.driver_env, policy, **args['rnn'])

    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['neptune']:
            path = NeptuneLogger(args, load_id, mode='read-only').download()
        elif args['wandb']:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError('No run id provided for eval')

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args['load_model_path']
    if load_path == 'latest':
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        #state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        #optim_state = torch.load(state_path)['optimizer_state_dict']
        #pufferl.optimizer.load_state_dict(optim_state)

    return policy

def load_config(env_name, parser=None):
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))

    return process_config(p, parser=parser)

def load_config_file(file_path, fill_in_default=True, parser=None):
    if not os.path.exists(file_path):
        raise pufferlib.APIUsageError('No config file found')

    config_paths = [file_path]

    if fill_in_default:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
        # Process the puffer defaults first
        config_paths.insert(0, os.path.join(puffer_dir, 'config/default.ini'))

    p = configparser.ConfigParser()
    p.read(config_paths)

    return process_config(p, parser=parser)

def make_parser():
    '''Creates the argument parser with default PufferLib arguments.'''
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandb/Neptune run')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--neptune', action='store_true', help='Use neptune for logging')
    parser.add_argument('--neptune-name', type=str, default='pufferai')
    parser.add_argument('--neptune-project', type=str, default='ablations')
    parser.add_argument('--log-dir', type=str, default='experiments', help='Root directory for logs and checkpoints')
    parser.add_argument('--tensorboard', action='store_true', help='Use TensorBoard for logging')
    parser.add_argument('--no-model-upload', action='store_true', help='Do not upload models to wandb or neptune')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    return parser

def process_config(config, parser=None):
    if parser is None:
        parser = make_parser()

    parser.description = f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    def auto_type(value):
        """Type inference for numeric args that use 'auto' as a default value"""
        if value == 'auto': return value
        if value.isnumeric(): return int(value)
        return float(value)

    for section in config.sections():
        for key in config[section]:
            try:
                value = ast.literal_eval(config[section][key])
            except:
                value = config[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=value,
                type=auto_type if value == 'auto' else type(value)
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args['train']['env'] = args['env_name'] or ''  # for trainer dashboard
    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args

def main():
    err = 'Usage: puffer [train, eval, sweep, autotune, profile, export] [env_name] [optional args]. --help for more info'
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == 'train':
        train(env_name=env_name)
    elif mode == 'eval':
        eval(env_name=env_name)
    elif mode == 'sweep':
        sweep(env_name=env_name)
    elif mode == 'autotune':
        autotune(env_name=env_name)
    elif mode == 'profile':
        profile(env_name=env_name)
    elif mode == 'export':
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)

if __name__ == '__main__':
    main()
