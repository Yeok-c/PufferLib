![figure](https://pufferai.github.io/source/resource/header.png)

[![PyPI version](https://badge.fury.io/py/pufferlib.svg)](https://badge.fury.io/py/pufferlib)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pufferlib)
![Github Actions](https://github.com/PufferAI/PufferLib/actions/workflows/install.yml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

PufferLib is the reinforcement learning library I wish existed during my PhD. It started as a compatibility layer to make working with complex environments a breeze. Now, it's a high-performance toolkit for research and industry with optimized parallel simulation, environments that run and train at 1M+ steps/second, and tons of quality of life improvements for practitioners. All our tools are free and open source. We also offer priority service for companies, startups, and labs!

![Trailer](https://github.com/PufferAI/puffer.ai/blob/main/docs/assets/puffer_2.gif?raw=true)

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support -- post here before opening issues. We're always looking for new contributors, too!

## Installation (Gymnasium 1.x baseline)

PufferLib now targets **Gymnasium 1.x**.

- **Python**: `>=3.10` (was `>=3.9`)
- **Gymnasium**: `>=1.2.2` (was `==0.29.1`)
- **Gym (OpenAI gym)**: phased out (not installed / supported by default)

Install the base package:

```bash
# uv
uv sync

# pip
pip install -e . --no-build-isolation
```

## Environments (extras)

Environment integrations live under `[project.optional-dependencies]` and are installed via “extras”.

- **uv**: `uv sync --extra <name>`
- **pip**: `pip install "pufferlib[<name>]"`

Currently supported extras:

- `atari`
- `box2d`
- `butterfly`
- `classic_control`
- `craftax`
- `kinetix`
- `magent`
- `minigrid`
- `mujoco`
- `open_spiel`
- `pokemon_red`
- `vizdoom`
- `ray`
- `cleanrl`

## `uv sync` and “non-exclusivity” of extras

`uv` uses **universal resolution** for lockfiles: it tries to ensure the project’s dependency *universe* is solvable across supported Python/platform splits. As a result, **extras can affect `uv sync` even when you didn’t ask to install them**, because they may still be considered during lock resolution.

To make this workable for a repo with many third-party environment stacks, we explicitly declare **mutually incompatible extras** under `[tool.uv].conflicts` in `pyproject.toml`. This tells `uv` that certain extras are *not meant to be installed together*, so they don’t have to resolve as a single combined environment.

Practical implications:

- If you only need the base library, run `uv sync` (no extras).
- If you need one environment stack, use `uv sync --extra <env>`.
- If you need multiple stacks that conflict (e.g. due to `pygame` / `nle` pins), use **separate virtual environments**.

## Phased-out environments (and why)

Some extras were removed/disabled because they prevented `uv` from producing a consistent lockfile under the Gymnasium 1.x baseline, or because upstream packages are not currently usable in a reproducible way:

- **`avalon`**: upstream `avalon-rl` pins `gym==0.25.2` which conflicts with PufferLib’s `gym==0.23` compatibility layer.
- **`metta` / `cogames`**: Git-based stacks with Python-version constraints (notably Python `>=3.12`) and/or packaging metadata that makes PEP 508 / lock resolution unreliable in this repo.
- **`tribal-village`**: upstream requires Python `>=3.12`.
- **`slimevolley`**: referenced package name/version is not available on PyPI (`slimevolley==0.1.0`).
- **`nmmo`**: upstream build currently fails (missing Cython sources in published sdist/wheel), which breaks locking on fresh environments.
- **`procgen`**: upstream `procgen-mirror` is pinned to `numpy<2.0.0` and currently only publishes `0.10.7`, making it incompatible with the NumPy 2.x baseline.
- **Gym-based stacks (`gym` / `shimmy`)**: `gym` is phased out under the NumPy 2.x + Gymnasium 1.x baseline, so integrations that rely on `gym` + `shimmy` are no longer supported as extras: `bsuite`, `crafter`, `dm_control`, `dm_lab`, `griddly`, `microrts`, `minerl`, `minihack`, `nethack`.

If you need one of these stacks, the recommended approach is to pin against an older PufferLib release line (pre Gymnasium 1.x baseline) or install the environment package stack manually in an isolated environment and integrate it locally.

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>