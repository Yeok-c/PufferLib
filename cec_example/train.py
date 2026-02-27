"""Standalone training script for DASM.

Uses PufferLib's PuffeRL training loop but creates the environment and
policy directly, bypassing pufferlib's config / env_creator / load_policy
pipeline.  The policy is loaded from src/policy.py so it can be modified
independently.

Usage:
    python src/train.py [--device cuda] [--total-timesteps 200000000] ...
    python src/train.py --help
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

import pufferlib
import pufferlib.vector
from pufferlib.pufferl import PuffeRL, NoLogger, WandbLogger
from pufferlib.ocean.dasm.dasm import Dasm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy import Policy


def parse_args():
    p = argparse.ArgumentParser(description="Train DASM")

    # -- Environment ----------------------------------------------------------
    env = p.add_argument_group("environment")
    env.add_argument("--num-envs", type=int, default=128)
    env.add_argument("--num-agents", type=int, default=2)
    env.add_argument("--num-blocks", type=int, default=5)
    env.add_argument("--arena-width", type=float, default=30.0)
    env.add_argument("--arena-height", type=float, default=20.0)
    env.add_argument("--gravity", type=float, default=9.81)
    env.add_argument("--goal-box-width", type=float, default=5.0)
    env.add_argument("--goal-box-height", type=float, default=2.5)
    env.add_argument("--block-min-size", type=float, default=0.5)
    env.add_argument("--block-max-size", type=float, default=1.2)
    env.add_argument("--agent-move-force", type=float, default=20.0)
    env.add_argument("--gripper-speed", type=float, default=3.0)
    env.add_argument("--grab-threshold", type=float, default=0.7)
    env.add_argument("--release-threshold", type=float, default=0.3)
    env.add_argument("--grab-distance", type=float, default=1.8)
    env.add_argument("--reward-block-placed", type=float, default=1.0)
    env.add_argument("--reward-all-placed", type=float, default=2.0)
    env.add_argument("--reward-step-penalty", type=float, default=-0.001)
    env.add_argument("--max-episode-steps", type=int, default=300)

    # -- Policy ---------------------------------------------------------------
    pol = p.add_argument_group("policy")
    pol.add_argument("--hidden-size", type=int, default=256)

    # -- Training -------------------------------------------------------------
    train = p.add_argument_group("training")
    train.add_argument("--device", type=str, default="cuda")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--total-timesteps", type=int, default=200_000_000)
    train.add_argument("--learning-rate", type=float, default=0.0003)
    train.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "muon"])
    train.add_argument("--gamma", type=float, default=0.995)
    train.add_argument("--gae-lambda", type=float, default=0.90)
    train.add_argument("--update-epochs", type=int, default=1)
    train.add_argument("--clip-coef", type=float, default=0.2)
    train.add_argument("--vf-coef", type=float, default=2.0)
    train.add_argument("--vf-clip-coef", type=float, default=0.2)
    train.add_argument("--max-grad-norm", type=float, default=1.5)
    train.add_argument("--ent-coef", type=float, default=0.001)
    train.add_argument("--batch-size", type=int, default=None,
                        help="Defaults to total_agents * bptt_horizon")
    train.add_argument("--minibatch-size", type=int, default=8192)
    train.add_argument("--max-minibatch-size", type=int, default=32768)
    train.add_argument("--bptt-horizon", type=int, default=64)
    train.add_argument("--checkpoint-interval", type=int, default=250)
    train.add_argument("--data-dir", type=str, default="experiments")

    # -- Advanced PPO ---------------------------------------------------------
    adv = p.add_argument_group("advanced")
    adv.add_argument("--prio-alpha", type=float, default=0.8)
    adv.add_argument("--prio-beta0", type=float, default=0.2)
    adv.add_argument("--vtrace-rho-clip", type=float, default=1.0)
    adv.add_argument("--vtrace-c-clip", type=float, default=1.0)
    adv.add_argument("--adam-beta1", type=float, default=0.95)
    adv.add_argument("--adam-beta2", type=float, default=0.999)
    adv.add_argument("--adam-eps", type=float, default=1e-12)

    # -- Checkpointing / resume -----------------------------------------------
    ckpt = p.add_argument_group("checkpoint")
    ckpt.add_argument("--load-model-path", type=str, default=None,
                       help='Path to .pt checkpoint, or "latest"')

    # -- Logging --------------------------------------------------------------
    log = p.add_argument_group("logging")
    log.add_argument("--wandb", action="store_true")
    log.add_argument("--wandb-project", type=str, default="dasm")
    log.add_argument("--wandb-group", type=str, default="debug")

    return p.parse_args()


def make_env(args):
    """Create a natively-vectorized DASM PufferEnv."""
    env_kwargs = dict(
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        num_blocks=args.num_blocks,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
        gravity=args.gravity,
        goal_box_width=args.goal_box_width,
        goal_box_height=args.goal_box_height,
        block_min_size=args.block_min_size,
        block_max_size=args.block_max_size,
        agent_move_force=args.agent_move_force,
        gripper_speed=args.gripper_speed,
        grab_threshold=args.grab_threshold,
        release_threshold=args.release_threshold,
        grab_distance=args.grab_distance,
        reward_block_placed=args.reward_block_placed,
        reward_all_placed=args.reward_all_placed,
        reward_step_penalty=args.reward_step_penalty,
        max_episode_steps=args.max_episode_steps,
    )
    return pufferlib.vector.make(Dasm, env_kwargs=env_kwargs)


def make_policy(env, args):
    """Instantiate the policy and optionally load a checkpoint."""
    policy = Policy(env.driver_env, hidden_size=args.hidden_size)
    policy = policy.to(args.device)

    load_path = args.load_model_path
    if load_path == "latest":
        candidates = glob.glob(os.path.join(args.data_dir, "dasm*/*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoints found in " + args.data_dir)
        load_path = max(candidates, key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=args.device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {load_path}")

    return policy


def build_train_config(args, total_agents):
    """Assemble the flat config dict that PuffeRL expects."""
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = total_agents * args.bptt_horizon

    return {
        "env": "dasm",
        "seed": args.seed,
        "torch_deterministic": True,
        "cpu_offload": False,
        "device": args.device,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "anneal_lr": True,
        "min_lr_ratio": 0.0,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "update_epochs": args.update_epochs,
        "clip_coef": args.clip_coef,
        "vf_coef": args.vf_coef,
        "vf_clip_coef": args.vf_clip_coef,
        "max_grad_norm": args.max_grad_norm,
        "ent_coef": args.ent_coef,
        "batch_size": batch_size,
        "minibatch_size": args.minibatch_size,
        "max_minibatch_size": args.max_minibatch_size,
        "bptt_horizon": args.bptt_horizon,
        "compile": False,
        "compile_mode": "reduce-overhead",
        "use_rnn": False,
        "optimizer": args.optimizer,
        "precision": "float32",
        "data_dir": args.data_dir,
        "checkpoint_interval": args.checkpoint_interval,
        "prio_alpha": args.prio_alpha,
        "prio_beta0": args.prio_beta0,
        "vtrace_rho_clip": args.vtrace_rho_clip,
        "vtrace_c_clip": args.vtrace_c_clip,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
    }


def main():
    args = parse_args()

    vecenv = make_env(args)
    policy = make_policy(vecenv, args)
    train_config = build_train_config(args, vecenv.num_agents)

    logger = NoLogger({})
    if args.wandb:
        logger = WandbLogger({
            "wandb_project": args.wandb_project,
            "wandb_group": args.wandb_group,
            "tag": None,
            "no_model_upload": False,
        })

    trainer = PuffeRL(train_config, vecenv, policy, logger)

    while trainer.global_step < train_config["total_timesteps"]:
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        trainer.evaluate()
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        trainer.train()

    model_path = trainer.close()
    logger.close(model_path, early_stop=False)
    print(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()
