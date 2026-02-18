import functools
import importlib

import gymnasium

import pufferlib.emulation
import pufferlib.environments

ALIASES = {
    'minigrid': 'minigrid-empty',
}
EMPTY_MIN_GRID_SIZE = 5
EMPTY_MAX_GRID_SIZE = 64
MAX_EPISODE_TICKS = 100


def env_creator(name='minigrid'):
    return functools.partial(make, name=name)

def make(
    name,
    render_mode='rgb_array',
    buf=None,
    seed=0,
    grid_size=7,
    agent_view_size=None,
    max_size=None,
    map_size=None,
):
    if name in ALIASES:
        name = ALIASES[name]

    pufferlib.environments.try_import('minigrid')

    if name in {'minigrid-empty', 'empty'}:
        minigrid_envs = importlib.import_module('minigrid.envs')

        # Accept ocean/grid-style kwargs, but keep Minigrid size controlled
        # by one value.
        if grid_size is None:
            if map_size is not None:
                grid_size = map_size
            elif max_size is not None:
                grid_size = max_size
            else:
                grid_size = 7

        grid_size = int(grid_size)
        max_size = grid_size
        map_size = grid_size
        if not (EMPTY_MIN_GRID_SIZE <= grid_size <= EMPTY_MAX_GRID_SIZE):
            raise ValueError(
                f'Unsupported minigrid grid_size={grid_size}. '
                f'Valid range: [{EMPTY_MIN_GRID_SIZE}, {EMPTY_MAX_GRID_SIZE}]'
            )
        if agent_view_size is None:
            # Minigrid requires odd view sizes. Match grid_size as closely as
            # possible while respecting that constraint.
            agent_view_size = grid_size if grid_size % 2 == 1 else grid_size - 1

        env_kwargs = {
            'size': grid_size,
            'render_mode': render_mode,
        }
        env_kwargs['agent_view_size'] = int(agent_view_size)
        env = minigrid_envs.EmptyEnv(**env_kwargs)
    else:
        env = gymnasium.make(name, render_mode=render_mode)

    env = MiniGridWrapper(env)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class MiniGridWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = gymnasium.spaces.Dict({
            k: v for k, v in self.env.observation_space.items() if
            k != 'mission'
        })
        self.action_space = self.env.action_space
        self.close = self.env.close
        self.render = self.env.render
        self.close = self.env.close
        self.render_mode = 'rgb_array'

    def reset(self, seed=None, options=None):
        self.tick = 0
        obs, info = self.env.reset(seed=seed)
        del obs['mission']
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        del obs['mission']

        self.tick += 1
        if self.tick == MAX_EPISODE_TICKS:
            done = True

        return obs, reward, done, truncated, info
