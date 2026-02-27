import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.racer_simple import binding


class RacerSimple(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 frameskip=2, width=1080, height=720,
                 track_width=75, max_whisker_length=100,
                 num_whiskers=10, w_ang=3.14159, turn_rate=0.0785,
                 maxv=5, min_v=1.0, accel=0.2, decel=0.3,
                 reward_scale=1.0, car_radius=18,
                 num_npcs=5, npc_radius=18,
                 max_lives=3, collision_penalty=1.0,
                 num_points=16, bezier_resolution=4,
                 log_interval=128, seed=42, buf=None,
                 rng=42, i=1):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(num_whiskers + 2,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)

        c_envs = []
        for i_env in range(num_envs):
            env_id = binding.env_init(
                self.observations[i_env:i_env+1],
                self.actions[i_env:i_env+1],
                self.rewards[i_env:i_env+1],
                self.terminals[i_env:i_env+1],
                self.truncations[i_env:i_env+1],
                seed,
                width=width, height=height,
                track_width=track_width,
                max_whisker_length=max_whisker_length,
                num_whiskers=num_whiskers,
                w_ang=w_ang, turn_rate=turn_rate,
                maxv=maxv, min_v=min_v,
                accel=accel, decel=decel,
                reward_scale=reward_scale,
                car_radius=car_radius,
                num_npcs=num_npcs, npc_radius=npc_radius,
                max_lives=max_lives,
                collision_penalty=collision_penalty,
                frameskip=frameskip,
                num_points=num_points,
                bezier_resolution=bezier_resolution,
                rng=rng + i_env, i=i_env,
            )
            c_envs.append(env_id)
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        return binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)
