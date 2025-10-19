## arc-agi

Pure-JAX ARC environment for RL and program synthesis experiments.

### Features

- **JAX-native**: Fully JIT-compilable step function with pytree-compatible state
- **Masked reward**: Only evaluates correctness on valid (non-padded) cells
- **Extended action space**: 18 actions including erase, crop, and shift-to-origin
- **Batching support**: Vmapped reset and step for parallel environments
- **Flexible observations**: Multi-channel grid observations with optional compact format

### Quickstart

Run demos to see all actions in action:

```bash
# Run all demos (basic, erase, crop, shift-to-origin)
python -m scripts.run_arc_env --max_steps 100

# Run specific demo with visualization
python -m scripts.run_arc_env --demo crop --render

# Run on your own ARC dataset
python -m scripts.run_arc_env --data_dir /path/to/arc/jsons --max_steps 100
```

### Action Space

The environment provides **18 discrete actions** (0-17):

| Action | Description |
|--------|-------------|
| 0-3    | Move cursor: up, down, left, right |
| 4-13   | Paint colors 0-9 at cursor position |
| 14     | Copy value from input grid at cursor to canvas |
| 15     | Erase (set cell to -1 padding value) |
| 16     | Crop: fill everything right and down from cursor with -1 |
| 17     | Move-to-origin: shift grid so cursor becomes (0,0), pad with color 0 |

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

- **Reward**: Incremental change in correctly matched cells (on valid, non-padded regions only)
- **Termination**: Episode ends when:
  - Step budget (`max_steps`) is exhausted, OR
  - All valid cells match the target (ignores -1 padded cells)

### Key Improvements

This version fixes several issues from the original:

1. **Masked evaluation**: Reward and success only consider non-padded cells
2. **Erase action**: Can now undo painting mistakes by setting cells to -1
3. **Empty dataset guards**: Clear errors when train/test sets are empty
4. **Action validation**: Actions are clamped to prevent out-of-bounds errors
5. **Crop & shift**: New high-level actions for canvas manipulation
6. **Type cleanup**: Removed unused `ARCState`, improved documentation
