# racer_simple

A minimal 2D racing environment built in C with Raylib rendering, designed for reinforcement learning via PufferLib. The agent drives a car around procedurally generated tracks, avoids walls and NPC traffic, and learns through whisker-based perception.

## Quick Start

```bash
# Train (from repo root, with venv active)
puffer train racer_simple --train.device mps   # or cpu/cuda

# Export weights and build standalone demo binary
cd src/PufferLib/pufferlib/ocean/racer_simple
bash build.sh        # debug build
bash build.sh fast   # optimized build

# Run the demo (from src/PufferLib/)
./racer_simple
```

If no valid weights file is found, the demo falls back to random actions. You can take manual control at any time by holding **Shift** and using arrow keys (Left/Right to steer, Down to brake; default action is accelerate).

## Features

### Procedural Track Generation
Each episode generates a new closed-loop track using randomized sinusoidal control points connected by cubic Bezier curves. The track is divided into inner and outer edges with a configurable width, producing varied curves every reset.

### 3-Lane NPC Traffic
The track is conceptually divided into 3 lanes (inner, center, outer). On each reset, a configurable number of NPC cars are spawned at random positions across random lanes. Each lane is assigned a random speed (slow relative to the player). NPCs travel along the track continuously, with smooth interpolation between track points.

### Whisker-Based Perception
The car perceives its surroundings through N configurable whisker rays (default 10) cast in all directions (full 360 degrees). Each whisker reports a normalized distance (0-1) to the nearest obstacle, which can be a wall segment or an NPC bounding circle. This is the agent's primary sensory input -- no images, no global state.

### Variable Speed with 4-Action Control
| Action    | Effect                                    |
|-----------|-------------------------------------------|
| `LEFT`    | Rotate heading counter-clockwise          |
| `FORWARD` | Accelerate (increase velocity up to maxv) |
| `RIGHT`   | Rotate heading clockwise                  |
| `BRAKE`   | Decelerate (decrease velocity to min_v)   |

The car always moves forward by its current velocity after the action is applied. Velocity persists between steps (inertia).

### Lives System
The player starts with 3 lives. Hitting an NPC car causes: velocity drops to 0, one life is lost, a score penalty is applied, and 30 ticks of invincibility (player flashes). When lives reach 0, the episode terminates. Hitting a wall is always instantly terminal regardless of lives.

### Reward Signal
Reward is the counter-clockwise polar angle delta from the track center. The agent is rewarded for making forward progress around the track. NPC collisions apply a negative penalty.

### Observation Space
`(num_whiskers + 2,)` floats, all normalized to [0, 1]:
```
[whisker_0, whisker_1, ..., whisker_N-1, velocity/maxv, lives/max_lives]
```

### HUD (Demo)
The demo window displays `Lives: X | Score: Y.Z` in the top-right corner. The player car is yellow, NPC cars are colored by lane (gray/red/blue), whiskers are colored on a blue-to-red gradient, and lane dividers are drawn as thin gray dashes.

## Architecture

### File Structure

```
racer_simple/
  racer_simple.h    -- Structs, macros, inline math helpers, function prototypes
  racer_simple.c    -- All game logic + rendering + standalone demo (ifdef RACER_SIMPLE_DEMO)
  binding.c         -- Python/C bindings via PufferLib's env_binding.h
  racer_simple.py   -- Python wrapper (gymnasium-compatible PufferEnv)
  build.sh          -- Finds trained weights, exports to .bin, compiles standalone binary
  README.md         -- This file

config/ocean/racer_simple.ini  -- All hyperparameters (env, policy, training)
```

### How Training Works

1. `puffer train racer_simple` reads `racer_simple.ini` for all config
2. PufferLib creates 1024 parallel C environments via the Python binding
3. Each env runs the game loop in C: `c_step()` applies the action for `frameskip` frames, each frame calling `step_frame()` which handles physics, NPC movement, whisker raycasting, collision detection, and reward computation
4. Observations (whisker lengths + velocity + lives) flow back to Python as a flat float array
5. A `LinearLSTM` policy network (encoder -> LSTM -> value/policy heads) processes observations and outputs actions
6. PPO training updates the policy using the collected trajectories
7. Checkpoints are saved periodically to `experiments/puffer_racer_simple_*/model_*.pt`

### How Export and Demo Work

`build.sh` does three things:
1. Finds the newest `.pt` checkpoint in `experiments/`
2. Runs a Python one-liner that loads the PyTorch state dict, filters out duplicate LSTM cell parameters, concatenates all weight tensors to a flat `.bin` file
3. Compiles `racer_simple.c` with `-DRACER_SIMPLE_DEMO` and links against Raylib to produce a standalone binary

The demo binary loads the `.bin` weights into PufferLib's C-native `LinearLSTM` inference engine (from `puffernet.h`) -- no Python or PyTorch at runtime. It runs the same game loop as training but with Raylib rendering at 60fps.

### Key C Functions

| Function | Purpose |
|---|---|
| `generate_track()` | Creates a random closed-loop Bezier track with inner/outer edges |
| `spawn_npcs()` | Places NPCs at random lane positions with random per-lane speeds |
| `update_npcs()` | Advances all NPCs along their lanes each frame |
| `step_frame()` | One physics tick: action -> velocity -> movement -> NPCs -> whiskers -> collision -> reward |
| `compute_whiskers()` | Raycasts N whiskers against wall segments and NPC circles |
| `check_npc_collision()` | Circle-circle overlap between player and NPCs; handles lives/invincibility |
| `compute_reward()` | Polar angle progress around the track center |
| `compute_observations()` | Packs whisker lengths + normalized velocity + normalized lives |
| `c_render()` | Raylib drawing: track, lane dividers, NPCs, player, whiskers, HUD |

### Configurable Parameters

All parameters are set in `racer_simple.ini` and passed through the Python wrapper to C:

| Parameter | Default | Description |
|---|---|---|
| `num_whiskers` | 10 | Number of whisker rays |
| `w_ang` | PI | Half-angle of whisker spread (PI = full 360) |
| `max_whisker_length` | 100 | Max raycast distance in pixels |
| `maxv` / `min_v` | 5 / 1.0 | Velocity bounds |
| `accel` / `decel` | 0.2 / 0.3 | Acceleration/deceleration per tick |
| `turn_rate` | 0.0785 | Radians turned per LEFT/RIGHT action |
| `num_npcs` | 5 | Number of NPC traffic cars |
| `npc_radius` | 18 | NPC bounding circle radius |
| `max_lives` | 3 | Lives before episode terminates |
| `collision_penalty` | 1.0 | Score/reward deducted per NPC hit |
| `track_width` | 75 | Road width in pixels |
| `frameskip` | 4 | Physics frames per agent step |
| `num_points` | 16 | Control points for track generation |
| `bezier_resolution` | 4 | Subdivisions per Bezier segment |
