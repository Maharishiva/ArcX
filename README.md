## arc-agi

Pure-JAX ARC environment for RL and program synthesis experiments, compatible with the official [ARC-AGI-2 dataset](https://github.com/arcprize/ARC-AGI-2).

### Features

- **JAX-native**: Fully JIT-compilable step function with pytree-compatible state
- **Rich reward system**: Compositional reward (IoU + valid-cell match + full match) with sparse/dense modes
- **Extended action space**: 20 actions including paint, flood fill, crop, and shift-to-origin
- **Batching support**: Vmapped reset and step for parallel environments
- **Flexible observations**: Multi-channel grid observations with cursor and action history
- **Official dataset**: Includes ARC-AGI-2 with 1,000 training and 400 evaluation tasks
- **PPO training**: Full RL pipeline with PerceiverActorCritic model (~4.8M params)

### Dataset

The `data/` directory contains the official ARC-AGI-2 dataset:
- **Training**: 1,000 tasks (3,232 train examples + 1,076 test examples) for model training
- **Evaluation**: 400 tasks for testing (120 public subset, human performance: 66% average)

Each task consists of demonstration input/output pairs and test cases where the goal is to produce correct outputs for all test inputs.

### Quickstart

**Environment demos:**
```bash
# Run demos with built-in sample task
python -m scripts.run_arc_env --demo all

# Run on the full ARC-AGI-2 training dataset
python -m scripts.run_arc_env --data_dir data/training --demo basic

# Run on evaluation set
python -m scripts.run_arc_env --data_dir data/evaluation --demo basic

# Visualize with rendering
python -m scripts.run_arc_env --data_dir data/training --demo shift --render
```

**PPO training:**
```bash
# Debug preset (quick smoke test)
PYTHONPATH=. python scripts/ppo_train.py --preset debug --total-updates 2

# A100 preset (full training on GPU)
PYTHONPATH=. python scripts/ppo_train.py --preset a100 --device cuda --total-updates 1000 --eval-interval 50

# Custom configuration
PYTHONPATH=. python scripts/ppo_train.py --num-envs 32 --rollout-length 64 --lr 3e-4
```

**Evaluate trained models:**
```bash
# Generate GIF visualizations of model rollouts
PYTHONPATH=. python scripts/ppo_eval_viz.py \
    --checkpoint checkpoints/ppo_a100/ppo_1000 \
    --num-episodes 4 \
    --rollout-horizon 64 \
    --output-dir artifacts/ppo_eval
```

See [`docs/colab_pro_plus.md`](docs/colab_pro_plus.md) for Colab setup and [`docs/ppo_training_overview.md`](docs/ppo_training_overview.md) for implementation details.

### Action Space

The environment provides **20 discrete actions** (0-19):

| Action | Description |
|--------|-------------|
| 0-3    | Move cursor: up, down, left, right |
| 4-13   | Choose paint color 0-9 (selected color persists) |
| 14     | Paint selected color at cursor position |
| 15     | Flood fill from cursor with selected color (4-neighborhood) |
| 16     | Crop: fill everything right and down from cursor with -1 |
| 17     | Move-to-origin: shift grid so cursor becomes (0,0), pad with color 0 |
| 18     | Send: terminate episode and evaluate reward |
| 19     | Copy entire input grid onto canvas |

### API

**Construction:**
- `ARCEnv.from_json(json_str, max_steps)` — construct from raw JSON string
- `data_loader.build_env_from_dir(dir_path, max_steps)` — load from folder of JSON files

**Core methods:**
- `env.env_reset(rng, train=True)` → `ARCEnvState`
- `env.env_step(state, action)` → `(next_state, reward, done)` (JIT-compiled)

**Observations:**
- `wrappers.observe(state, include_done_channel=True)` → `(4 or 5, 30, 30)` tensor
  - Channels: canvas, input, target, cursor_mask, [optional done_mask]
- `wrappers.observe_compact(state)` → dict with 'grid', 'cursor', 'done', 'steps'

**Batching:**
- `wrappers.make_batched_reset(env)` — vmapped reset over batch of RNG keys
- `wrappers.make_batched_step(env)` — vmapped step over batch of states/actions

### Reward and Termination

- **Reward**: Compositional reward with three components:
  - Valid-cell match bonus (0.5 if all non-padded cells match target)
  - IoU score (intersection over union of canvas and target valid regions)
  - Full match bonus (1.0 if entire grids match exactly)
- **Reward modes**: 
  - `sparse` (default): reward only on send/timeout
  - `dense`: reward on every step
- **Termination**: Episode ends when:
  - Send action (18) is taken, OR
  - Step budget (`max_steps`) is exhausted

### Architecture

**PerceiverActorCritic** (~4.8M parameters):
- **GridEncoder**: Processes task demonstrations and canvas with positional encodings
- **PerceiverIO**: Cross-attention encoder (128 latents × 4 layers) with cached demonstration embeddings
- **Policy head**: Per-canvas-cell action logits (20 actions) sampled at cursor position
- **Value head**: Single scalar value estimate from latent aggregation

Key design choices:
- Demonstration caching: encode input/target pairs once per rollout, reuse across steps
- Cursor-targeted policy: extract action logits for the current cursor cell only
- Full state conditioning: last action, selected color, time remaining, cursor mask

### Training Infrastructure

- **PPO implementation**: GAE-based advantage estimation, clipped surrogate objective
- **Presets**: `debug` (smoke test), `a100` (64 envs × 128 steps, 1e-4 LR)
- **Checkpointing**: Automatic saves to `checkpoints/ppo/` with Orbax
- **Evaluation**: Train/test split metrics, GIF generation for qualitative analysis
- **Device support**: CPU, CUDA, TPU via `--device` flag

See [`scripts/ppo_train.py`](scripts/ppo_train.py) for the full training pipeline.
