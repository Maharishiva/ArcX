# PR: Multi-IO Perceiver Context + Efficient Rollout Buffer + Progress-Shaped Reward

## Summary

This PR implements three major improvements to the ARC-AGI RL training pipeline:

1. **Multi-demo context**: Use ALL train input/output pairs (up to 5) from each problem as Perceiver cross-attention context, not just the current pair
2. **Efficient rollout buffer**: Remove per-step duplication of inputs/targets (~1.8MB saved per rollout); cache demo tokens once per env
3. **Progress-shaped reward**: Baseline-normalized reward that penalizes "just copy input" solutions and rewards stepwise progress

## Changes by Component

### Data Loading (`env/data_loader.py`)
- `load_arc_dir()` now returns problem grouping metadata: `train_group_starts` and `train_group_sizes`
- `build_env_from_dir()` accepts `max_demo_pairs` parameter and passes grouping to ARCEnv

### Environment (`env/env.py`, `env/types.py`)
- **ARCEnvState** extended with:
  - `baseline_score`: score of input grid vs target (for progress shaping)
  - `prev_progress`: `max(score_{t-1} - baseline, 0)` for reward delta computation
- **ARCEnv** extended with:
  - `train_group_starts`, `train_group_sizes`, `max_demo_pairs` fields
  - `build_context_batch(episode_idxs, train)`: Builds (task_grids, grid_type_ids, token_mask) for up to 5 train pairs per problem, with padding and masking
  - `_compute_score(canvas, target, valid_mask)`: Computes `0.2*IoU + valid_accuracy + 1.0*full_match`
- **Reward shaping**:
  - On reset: compute `baseline = score(input)` and `prev_progress = max(score(empty) - baseline, 0)`
  - On step: compute `progress_t = max(score_t - baseline, 0)` and `reward = progress_t - prev_progress`
  - This encourages improvement over input baseline and rewards stepwise progress

### Neural Network (`nets/`)
- **`transformer_utils.py`**: `PerceiverEncoderBlock` accepts optional `input_mask` param and passes to cross-attention
- **`PerceiverActorCritic.py`**: Accepts optional `cached_task_mask` param and reshapes to `(B, 1, 1, N_tokens)` for attention broadcasting

### Training (`scripts/ppo_train.py`)
- **RolloutBatch**: Removed `inputs`, `targets`; added `env_id` to map each row back to its environment
- **TaskCache**: Added `mask` field for token masking
- **`make_task_cache_fn`**: Uses `env.build_context_batch(episode_idxs)` to build multi-demo context once per env
- **`make_rollout_fn`**: Removed task_grids/grid_ids params; rollout stores `env_id` per step
- **`make_value_fn`**: Uses cached tokens/pos/mask directly (dummy grids for API compatibility)
- **`make_ppo_update_fn`**: Loss function gathers cached tokens/pos/mask per sample using `env_id`, avoiding per-step duplication
- **Training loop**: Cache demos once per rollout window, broadcast when flattening batch

### Evaluation (`scripts/ppo_eval_viz.py`)
- Updated policy functions to use cached context with mask
- Removed task_grids/grid_ids reconstruction

### Tests (`scripts/test_env.py`)
- Rewrote reward tests to reflect progress-shaped semantics
- Tests verify that painting correct cells gives positive progress reward
- Tests verify that copying input when input==output gives ~0 reward (baseline already at max)

### Documentation (`README.md`)
- Updated Features section to highlight progress-shaped rewards and multi-demo context
- Updated Reward and Termination section with full mathematical description of progress shaping
- Updated Architecture section to document multi-demo context, attention masking, and caching

## Memory Savings

With default config (num_envs=64, rollout_length=128):
- **Before**: inputs/targets stored at every step = 64 × 128 × 2 × 30 × 30 × 4 bytes = ~18.7 MB
- **After**: env_id stored at every step = 64 × 128 × 4 bytes = ~32 KB
- **Savings**: ~18.7 MB per rollout (~99.8% reduction in demonstration storage overhead)

Demo tokens are cached once per env (64 envs × ~9K tokens × 4 bytes × 128 latent_dim ≈ 295 MB) and reused across all 128 steps, eliminating redundant encoding.

## Backward Compatibility

- Default `max_demo_pairs=5` activates new behavior
- Setting `max_demo_pairs=1` emulates old 2-grid context (though reward is still progress-shaped)
- Tests updated and all passing

## Testing

- All 11 environment tests pass
- Progress-shaped reward verified with baseline normalization
- Termination conditions verified
- Multi-action sequences tested

## Migration Notes

If loading old checkpoints, note that:
1. ARCEnvState now has 2 additional fields (baseline_score, prev_progress)
2. Reward values will differ due to progress shaping
3. Model forward pass now accepts cached_task_mask (backward compatible if None)


