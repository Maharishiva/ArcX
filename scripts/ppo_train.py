from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NamedTuple, Tuple, Optional


def _select_device_from_cli(default: str = "auto") -> str:
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return os.getenv("ARC_JAX_DEVICE", default)


_requested_device = _select_device_from_cli()
if _requested_device and _requested_device.lower() != "auto":
    os.environ.setdefault("JAX_PLATFORM_NAME", _requested_device)
    os.environ.setdefault("JAX_PLATFORMS", _requested_device)

import jax.numpy as jnp
import jax
from jax import tree_util
import optax
from flax.training.train_state import TrainState
from flax.training import checkpoints

from env.data_loader import build_env_from_dir
from env.types import ARCEnvState
from nets.PerceiverActorCritic import PerceiverActorCritic


class RolloutBatch(NamedTuple):
    canvas: jnp.ndarray
    cursor: jnp.ndarray
    last_action: jnp.ndarray
    selected_color: jnp.ndarray
    steps: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    env_id: jnp.ndarray


class TaskCache(NamedTuple):
    grids: jnp.ndarray
    grid_ids: jnp.ndarray
    latents: jnp.ndarray
    mask: jnp.ndarray


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    data_dir: str = "data/training"
    num_envs: int = 8
    rollout_length: int = 32
    total_updates: int = 200
    learning_rate: float = 3e-4
    gamma: float = 0.996
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    num_minibatches: int = 4
    num_epochs: int = 4
    max_grad_norm: float = 0.5
    eval_envs: int = 8
    eval_horizon: int = 128
    eval_interval: int = 10
    log_interval: int = 1
    checkpoint_interval: int | None = None
    checkpoint_dir: str | None = "checkpoints/ppo"
    reward_mode: str = "dense"
    max_demo_pairs: int = 5
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "disabled"
    wandb_tags: Optional[Tuple[str, ...]] = None
def build_extra_canvas_features(
    cursor: jnp.ndarray,
    last_action: jnp.ndarray,
    selected_color: jnp.ndarray,
    steps: jnp.ndarray,
    grid_size: int,
    num_actions: int,
    max_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Construct per-cell canvas features plus cursor-token conditioning."""
    cursor_y = jnp.clip(cursor[:, 0], 0, grid_size - 1)
    cursor_x = jnp.clip(cursor[:, 1], 0, grid_size - 1)

    cursor_y_one = jax.nn.one_hot(cursor_y, grid_size, dtype=jnp.float32)
    cursor_x_one = jax.nn.one_hot(cursor_x, grid_size, dtype=jnp.float32)
    cursor_mask = jnp.einsum("bi,bj->bij", cursor_y_one, cursor_x_one)
    cursor_mask = cursor_mask[..., None]

    last_action_clipped = jnp.clip(last_action, 0, num_actions - 1)
    last_action_one_hot = jax.nn.one_hot(last_action_clipped, num_actions, dtype=jnp.float32)
    valid_last_action = (last_action >= 0).astype(jnp.float32)
    last_action_one_hot = last_action_one_hot * valid_last_action[:, None]
    last_action_feat = jnp.broadcast_to(
        last_action_one_hot[:, None, None, :],
        (cursor.shape[0], grid_size, grid_size, num_actions),
    )

    selected_color_feat = selected_color.astype(jnp.float32) / 9.0
    selected_color_feat = jnp.broadcast_to(
        selected_color_feat[:, None, None, None],
        (cursor.shape[0], grid_size, grid_size, 1),
    )

    time_remaining = 1.0 - (steps.astype(jnp.float32) / float(max_steps))
    time_remaining = jnp.clip(time_remaining, 0.0, 1.0)
    time_feat = jnp.broadcast_to(
        time_remaining[:, None, None, None],
        (cursor.shape[0], grid_size, grid_size, 1),
    )

    per_cell = jnp.concatenate([cursor_mask, last_action_feat, selected_color_feat, time_feat], axis=-1)

    cursor_norm = jnp.stack([cursor_y, cursor_x], axis=-1).astype(jnp.float32)
    cursor_norm = cursor_norm / jnp.maximum(grid_size - 1.0, 1.0)
    cursor_token = jnp.concatenate(
        [
            cursor_norm,
            last_action_one_hot,
            selected_color[:, None].astype(jnp.float32) / 9.0,
            time_remaining[:, None],
        ],
        axis=-1,
    )

    cursor_pos = jnp.stack([cursor_y.astype(jnp.float32), cursor_x.astype(jnp.float32)], axis=-1)
    return per_cell, cursor_token, cursor_pos


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dones_f = dones.astype(jnp.float32)
    next_values = jnp.concatenate([values[1:], last_value[None, ...]], axis=0)

    def step(carry, data):
        gae = carry
        reward, value, next_value, done = data
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return gae, gae

    reversed_data = [jnp.flip(d, axis=0) for d in (rewards, values, next_values, dones_f)]
    init_gae = jnp.zeros_like(last_value, dtype=jnp.float32)
    _, advantages_rev = jax.lax.scan(step, init_gae, reversed_data)
    advantages = jnp.flip(advantages_rev, axis=0)
    returns = advantages + values
    return advantages, returns


def flatten_batch(x: jnp.ndarray) -> jnp.ndarray:
    leading = x.shape[:2]
    rest = x.shape[2:]
    new_shape = (leading[0] * leading[1],) + rest
    return x.reshape(new_shape)


def downsample_token_mask(mask: jnp.ndarray, grid_size: int, patch_size: int) -> jnp.ndarray:
    if patch_size <= 1:
        return mask
    cells_per_grid = grid_size * grid_size
    num_grids = mask.shape[-1] // cells_per_grid
    rows = grid_size // patch_size
    cols = grid_size // patch_size
    reshaped = mask.reshape(mask.shape[0], 1, 1, num_grids, rows, patch_size, cols, patch_size)
    reshaped = jnp.any(reshaped, axis=-1)
    reshaped = jnp.any(reshaped, axis=-2)
    return reshaped.reshape(mask.shape[0], 1, 1, num_grids * rows * cols)


def make_task_cache_fn(model_apply, env, task_patch_size: int):
    @jax.jit
    def cache_fn(params: Dict[str, jnp.ndarray], problem_indices: jnp.ndarray):
        task_grids, grid_ids, token_mask = env.build_context_batch(problem_indices)
        patch_mask = downsample_token_mask(token_mask, env.GRID_SIZE, task_patch_size)
        latents = model_apply(
            {"params": params},
            task_grids,
            grid_ids,
            task_mask=patch_mask,
            deterministic=True,
            method=PerceiverActorCritic.prepare_task_latents,
        )
        return TaskCache(task_grids, grid_ids, latents, patch_mask)

    return cache_fn


def make_rollout_fn(
    model_apply,
    env,
    config: PPOConfig,
    grid_size: int,
    num_actions: int,
    max_steps: int,
) -> Tuple:
    def policy_forward(
        params: Dict[str, jnp.ndarray],
        state: ARCEnvState,
        cache: TaskCache,
    ):
        grid_feats, cursor_feats, cursor_pos = build_extra_canvas_features(
            state.cursor,
            state.last_action,
            state.selected_color,
            state.steps,
            grid_size,
            num_actions,
            max_steps,
        )
        cached_latents = jax.lax.stop_gradient(cache.latents)
        outputs = model_apply(
            {"params": params},
            cache.grids,
            cache.grid_ids,
            state.canvas,
            grid_feats,
            cursor_token_feats=cursor_feats,
            cursor_positions=cursor_pos,
            cached_task_latents=cached_latents,
            deterministic=True,
        )
        log_probs_all = jax.nn.log_softmax(outputs["logits"], axis=-1)
        return log_probs_all, outputs["value"]

    @jax.jit
    def policy_step(
        params: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
        state: ARCEnvState,
        cache: TaskCache,
    ):
        log_probs_all, values = policy_forward(params, state, cache)
        actions = jax.random.categorical(rng_key, log_probs_all, axis=-1).astype(jnp.int32)
        idx = jnp.arange(actions.shape[0])
        log_probs = log_probs_all[idx, actions]
        return actions, log_probs, values

    @jax.jit
    def rollout(
        params: Dict[str, jnp.ndarray],
        state: ARCEnvState,
        rng: jax.Array,
        cache: TaskCache,
    ):
        env_ids = jnp.arange(state.canvas.shape[0], dtype=jnp.int32)

        def step(carry, _):
            key, env_state = carry
            key, step_key = jax.random.split(key)
            actions, log_probs, values = policy_step(params, step_key, env_state, cache)
            next_state, rewards, dones = env.env_step_batch(env_state, actions)
            transition = (
                env_state.canvas,
                env_state.cursor,
                env_state.last_action,
                env_state.selected_color,
                env_state.steps,
                actions,
                log_probs,
                values,
                rewards,
                dones,
                env_ids,
            )
            return (key, next_state), transition

        (final_key, final_state), trajectory = jax.lax.scan(
            step,
            (rng, state),
            None,
            length=config.rollout_length,
        )
        return final_state, RolloutBatch(*trajectory), final_key

    return rollout, policy_step


def make_value_fn(
    model_apply,
    grid_size: int,
    num_actions: int,
    max_steps: int,
):
    @jax.jit
    def value_fn(
        params: Dict[str, jnp.ndarray],
        state: ARCEnvState,
        cache: TaskCache,
    ) -> jnp.ndarray:
        grid_feats, cursor_feats, cursor_pos = build_extra_canvas_features(
            state.cursor,
            state.last_action,
            state.selected_color,
            state.steps,
            grid_size,
            num_actions,
            max_steps,
        )
        cached_latents = jax.lax.stop_gradient(cache.latents)
        outputs = model_apply(
            {"params": params},
            cache.grids,
            cache.grid_ids,
            state.canvas,
            grid_feats,
            cursor_token_feats=cursor_feats,
            cursor_positions=cursor_pos,
            cached_task_latents=cached_latents,
            deterministic=True,
        )
        return outputs["value"]

    return value_fn


def make_ppo_update_fn(
    model_apply,
    config: PPOConfig,
    grid_size: int,
    num_actions: int,
    max_steps: int,
):
    batch_size = config.num_envs * config.rollout_length
    minibatch_size = batch_size // config.num_minibatches
    if minibatch_size % config.num_envs != 0:
        raise ValueError(
            "num_minibatches must divide rollout_length so that each minibatch spans an integer number of timesteps per environment."
        )

    steps_per_minibatch = minibatch_size // config.num_envs
    if steps_per_minibatch == 0:
        raise ValueError("num_minibatches cannot exceed rollout_length when avoiding cache duplication.")

    def loss_fn(params, minibatch, cache: TaskCache):
        metric_keys = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")

        task_latents = model_apply(
            {"params": params},
            cache.grids,
            cache.grid_ids,
            task_mask=cache.mask,
            deterministic=False,
            method=PerceiverActorCritic.prepare_task_latents,
        )

        flat_canvas = minibatch["canvas"].reshape(-1, grid_size, grid_size)
        flat_cursor = minibatch["cursor"].reshape(-1, 2)
        flat_last_action = minibatch["last_action"].reshape(-1)
        flat_selected = minibatch["selected_color"].reshape(-1)
        flat_steps = minibatch["steps"].reshape(-1)
        flat_actions = minibatch["actions"].reshape(-1)
        flat_old_log_probs = minibatch["old_log_probs"].reshape(-1)
        flat_advantages = minibatch["advantages"].reshape(-1)
        flat_returns = minibatch["returns"].reshape(-1)
        env_ids = minibatch["env_id"].reshape(-1)

        grid_feats, cursor_feats, cursor_pos = build_extra_canvas_features(
            flat_cursor,
            flat_last_action,
            flat_selected,
            flat_steps,
            grid_size,
            num_actions,
            max_steps,
        )

        sample_latents = jnp.take(task_latents, env_ids, axis=0)
        outputs = model_apply(
            {"params": params},
            None,
            None,
            flat_canvas,
            grid_feats,
            cursor_token_feats=cursor_feats,
            cursor_positions=cursor_pos,
            cached_task_latents=sample_latents,
            deterministic=False,
        )

        logits = outputs["logits"]
        values = outputs["value"].reshape(flat_returns.shape)
        log_probs_all = jax.nn.log_softmax(logits, axis=-1)
        idx = jnp.arange(logits.shape[0])
        new_log_probs = log_probs_all[idx, flat_actions]

        ratios = jnp.exp(new_log_probs - flat_old_log_probs)
        clipped = jnp.clip(ratios, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        policy_loss = -jnp.mean(jnp.minimum(ratios * flat_advantages, clipped * flat_advantages))

        value_loss = 0.5 * jnp.mean((values - flat_returns) ** 2)
        entropy = -jnp.mean(jnp.sum(jnp.exp(log_probs_all) * log_probs_all, axis=-1))
        approx_kl = jnp.mean(flat_old_log_probs - new_log_probs)
        clip_fraction = jnp.mean((jnp.abs(ratios - 1.0) > config.clip_epsilon).astype(jnp.float32))

        total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        metrics = {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        return total_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def update(train_state: TrainState, batch: Dict[str, jnp.ndarray], rng: jax.Array, cache: TaskCache):
        rollout_length = batch["advantages"].shape[0]

        def epoch_step(carry, _):
            state, key = carry
            key, perm_key = jax.random.split(key)
            permutation = jax.random.permutation(perm_key, rollout_length)
            permuted = permutation.reshape((config.num_minibatches, steps_per_minibatch))

            def minibatch_step(state, timestep_idx):
                minibatch = {k: jnp.take(v, timestep_idx, axis=0) for k, v in batch.items()}
                (loss_val, metrics), grads = grad_fn(state.params, minibatch, cache)
                state = state.apply_gradients(grads=grads)
                return state, metrics

            state, mb_metrics = jax.lax.scan(minibatch_step, state, permuted)
            metrics_avg = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), mb_metrics)
            return (state, key), metrics_avg

        (new_state, new_key), epoch_metrics = jax.lax.scan(
            epoch_step, (train_state, rng), jnp.arange(config.num_epochs)
        )
        metrics = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), epoch_metrics)
        return new_state, new_key, metrics

    return update


def reset_done_envs(env, state: ARCEnvState, rng: jax.Array, batch_size: int) -> ARCEnvState:
    reset_state = env.env_reset_batch(rng, train=True, batch_size=batch_size)
    done_mask = state.done

    def _select(new_field, old_field):
        mask = done_mask
        while mask.ndim < new_field.ndim:
            mask = mask[..., None]
        return jnp.where(mask, new_field, old_field)

    return ARCEnvState(
        rng=_select(reset_state.rng, state.rng),
        canvas=_select(reset_state.canvas, state.canvas),
        cursor=_select(reset_state.cursor, state.cursor),
        inp=_select(reset_state.inp, state.inp),
        target=_select(reset_state.target, state.target),
        valid_mask=_select(reset_state.valid_mask, state.valid_mask),
        done=_select(reset_state.done, state.done),
        steps=_select(reset_state.steps, state.steps),
        episode_idx=_select(reset_state.episode_idx, state.episode_idx),
        problem_idx=_select(reset_state.problem_idx, state.problem_idx),
        selected_color=_select(reset_state.selected_color, state.selected_color),
        last_action=_select(reset_state.last_action, state.last_action),
        baseline_score=_select(reset_state.baseline_score, state.baseline_score),
        prev_progress=_select(reset_state.prev_progress, state.prev_progress),
    )


def summarize_env_batch(env, state: ARCEnvState, episode_returns: jnp.ndarray) -> Dict[str, float]:
    returns = jnp.asarray(episode_returns, dtype=jnp.float32)
    steps = state.steps.astype(jnp.float32)
    done = state.done.astype(jnp.float32)

    success = jnp.all(state.canvas == state.target, axis=(1, 2)).astype(jnp.float32)
    solved = success * done
    solved_count = jnp.sum(done)
    solved_steps = jnp.sum(steps * done)
    mean_solution_steps = jnp.where(solved_count > 0.0, solved_steps / solved_count, 0.0)

    canvas_mask = state.canvas != env.EMPTY_CELL
    valid_mask = state.valid_mask
    intersection = jnp.sum((canvas_mask & valid_mask).astype(jnp.float32), axis=(1, 2))
    union = jnp.sum((canvas_mask | valid_mask).astype(jnp.float32), axis=(1, 2))
    mean_iou = jnp.mean(jnp.where(union > 0.0, intersection / union, jnp.ones_like(union)))

    valid_correct = jnp.where(valid_mask, state.canvas == state.target, True)
    valid_accuracy = jnp.mean(valid_correct.astype(jnp.float32))

    summary = {
        "return_mean": jnp.mean(returns),
        "return_std": jnp.std(returns),
        "return_max": jnp.max(returns),
        "done_rate": jnp.mean(done),
        "success_rate": jnp.mean(solved),
        "mean_steps": jnp.mean(steps),
        "mean_solution_steps": mean_solution_steps,
        "mean_iou": mean_iou,
        "valid_accuracy": valid_accuracy,
    }
    return {k: float(v) for k, v in summary.items()}


def evaluate_policy(
    params: Dict[str, jnp.ndarray],
    env,
    policy_step_sample,
    task_cache_fn,
    config: PPOConfig,
    rng: jax.Array,
) -> Tuple[jax.Array, Dict[str, float]]:
    def run_split(rng_key: jax.Array, train_flag: bool):
        rng_key, reset_key = jax.random.split(rng_key)
        state = env.env_reset_batch(reset_key, train=train_flag, batch_size=config.eval_envs)
        cache = task_cache_fn(params, state.problem_idx)
        returns = jnp.zeros((config.eval_envs,), dtype=jnp.float32)

        for _ in range(config.eval_horizon):
            rng_key, step_key = jax.random.split(rng_key)
            actions, _, _ = policy_step_sample(params, step_key, state, cache)
            state, rewards, dones = env.env_step_batch(state, actions)
            returns = returns + rewards
            if bool(jnp.all(dones)):
                break

        metrics = summarize_env_batch(env, state, returns)
        return rng_key, metrics

    rng, train_metrics = run_split(rng, train_flag=True)
    rng, test_metrics = run_split(rng, train_flag=False)

    eval_metrics = {
        "train_return": train_metrics["return_mean"],
        "train_return_std": train_metrics["return_std"],
        "train_return_max": train_metrics["return_max"],
        "train_done_rate": train_metrics["done_rate"],
        "train_success_rate": train_metrics["success_rate"],
        "train_mean_steps": train_metrics["mean_steps"],
        "train_mean_solution_steps": train_metrics["mean_solution_steps"],
        "train_mean_iou": train_metrics["mean_iou"],
        "train_valid_accuracy": train_metrics["valid_accuracy"],
        "test_return": test_metrics["return_mean"],
        "test_return_std": test_metrics["return_std"],
        "test_return_max": test_metrics["return_max"],
        "test_done_rate": test_metrics["done_rate"],
        "test_success_rate": test_metrics["success_rate"],
        "test_mean_steps": test_metrics["mean_steps"],
        "test_mean_solution_steps": test_metrics["mean_solution_steps"],
        "test_mean_iou": test_metrics["mean_iou"],
        "test_valid_accuracy": test_metrics["valid_accuracy"],
    }

    return rng, eval_metrics


def maybe_save_checkpoint(ckpt_dir: Path, step: int, train_state: TrainState):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir,
        target=train_state,
        step=step,
        overwrite=True,
        prefix="ppo_",
    )


def parse_args() -> tuple[PPOConfig, str]:
    parser = argparse.ArgumentParser(description="Train PPO agent on ARC-AGI tasks.")
    parser.add_argument("--preset", type=str, choices=["debug", "a100"], default=None, help="Preset hyperparameter bundle.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "tpu", "auto"], default=None, help="Device override for JAX runtime.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--reward-mode", type=str, choices=["sparse", "dense"], default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--eval-envs", type=int, default=None)
    parser.add_argument("--eval-horizon", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--max-demo-pairs", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, choices=["disabled", "online", "offline"], default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated list of tags for wandb runs.")
    args = parser.parse_args()

    config = PPOConfig()

    if (args.preset or "").lower() == "a100":
        config = dataclasses.replace(
            config,
            num_envs=64,
            rollout_length=128,
            total_updates=5000,
            learning_rate=1e-4,
            num_minibatches=16,
            num_epochs=2,
            eval_envs=32,
            eval_horizon=256,
            eval_interval=100,
            checkpoint_dir="checkpoints/ppo_a100",
        )

    field_names = {field.name for field in dataclasses.fields(PPOConfig)}
    wandb_tags_parsed: Optional[Tuple[str, ...]] = None
    if args.wandb_tags:
        tags = [tag.strip() for tag in args.wandb_tags.split(",")]
        tags = [tag for tag in tags if tag]
        if tags:
            wandb_tags_parsed = tuple(tags)

    overrides = {}
    for key, value in vars(args).items():
        if key == "wandb_tags":
            continue
        if key in field_names and value is not None:
            overrides[key] = value

    if overrides:
        config = dataclasses.replace(config, **overrides)

    if wandb_tags_parsed is not None:
        config = dataclasses.replace(config, wandb_tags=wandb_tags_parsed)

    if config.max_grad_norm is not None and config.max_grad_norm < 0:
        config = dataclasses.replace(config, max_grad_norm=None)

    active_device = args.device or _requested_device or "cpu"
    return config, active_device


def main():
    config, device_choice = parse_args()
    use_wandb = config.wandb_mode != "disabled"
    wandb_module = None
    wandb_run = None
    if use_wandb:
        if not config.wandb_project:
            raise ValueError("wandb_project must be provided when wandb logging is enabled.")
        try:
            import wandb as _wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("wandb is required for logging; install wandb or disable wandb logging.") from exc

        wandb_module = _wandb
        wandb_init_kwargs = {
            "project": config.wandb_project,
            "entity": config.wandb_entity,
            "name": config.wandb_run_name,
            "mode": config.wandb_mode,
            "config": dataclasses.asdict(config),
        }
        if config.wandb_tags:
            wandb_init_kwargs["tags"] = list(config.wandb_tags)
        # Drop None values so wandb.init ignores them
        wandb_init_kwargs = {k: v for k, v in wandb_init_kwargs.items() if v is not None}
        wandb_run = wandb_module.init(**wandb_init_kwargs)
    assert (config.num_envs * config.rollout_length) % config.num_minibatches == 0, "Minibatch size must divide rollout size"

    try:
        cpu_devices = jax.devices("cpu")
        load_ctx = jax.default_device(cpu_devices[0]) if cpu_devices else nullcontext()
    except RuntimeError:
        load_ctx = nullcontext()
    with load_ctx:
        env = build_env_from_dir(
            Path(config.data_dir),
            reward_mode=config.reward_mode,
            max_demo_pairs=config.max_demo_pairs,
        )
    grid_size = env.GRID_SIZE
    num_actions = env.NUM_ACTIONS
    max_steps = env.max_steps

    model = PerceiverActorCritic(dtype=jnp.bfloat16, use_remat=True)

    rng = jax.random.PRNGKey(config.seed)
    rng, reset_key = jax.random.split(rng)
    state = env.env_reset_batch(reset_key, train=True, batch_size=config.num_envs)

    task_grids, grid_ids, init_mask = env.build_context_batch(state.problem_idx)
    init_mask_patch = downsample_token_mask(init_mask, grid_size, model.task_patch_size)

    init_grid_feats, init_cursor_feats, init_cursor_pos = build_extra_canvas_features(
        state.cursor,
        state.last_action,
        state.selected_color,
        state.steps,
        grid_size,
        num_actions,
        max_steps,
    )

    rng, init_key = jax.random.split(rng)
    params = model.init(
        init_key,
        task_grids,
        grid_ids,
        state.canvas,
        init_grid_feats,
        cursor_token_feats=init_cursor_feats,
        cursor_positions=init_cursor_pos,
        task_mask=init_mask_patch,
    )

    if config.max_grad_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    else:
        tx = optax.adam(config.learning_rate)
    train_state = TrainState.create(apply_fn=model.apply, params=params["params"], tx=tx)

    rollout_fn, policy_step_fn = make_rollout_fn(model.apply, env, config, grid_size, num_actions, max_steps)
    value_fn = make_value_fn(model.apply, grid_size, num_actions, max_steps)
    ppo_update_fn = make_ppo_update_fn(model.apply, config, grid_size, num_actions, max_steps)
    task_cache_fn = make_task_cache_fn(model.apply, env, model.task_patch_size)

    param_count = sum(int(x.size) for x in tree_util.tree_leaves(train_state.params))
    print(
        f"Initialized PerceiverActorCritic with {param_count / 1e6:.2f}M trainable parameters "
        f"on device preset '{device_choice}'."
    )

    ckpt_dir = Path(config.checkpoint_dir).resolve() if config.checkpoint_dir is not None else None

    for update_idx in range(config.total_updates):
        rng, rollout_key = jax.random.split(rng)
        task_cache = task_cache_fn(train_state.params, state.problem_idx)
        next_state, traj, rollout_key = rollout_fn(
            train_state.params,
            state,
            rollout_key,
            task_cache,
        )

        next_task_cache = task_cache_fn(train_state.params, next_state.problem_idx)
        last_value = value_fn(
            train_state.params,
            next_state,
            next_task_cache,
        )
        advantages, returns = compute_gae(
            traj.rewards,
            traj.values,
            traj.dones,
            last_value,
            config.gamma,
            config.gae_lambda,
        )

        adv_mean = jnp.mean(advantages)
        adv_std = jnp.std(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        batch = {
            "canvas": traj.canvas,
            "cursor": traj.cursor,
            "last_action": traj.last_action,
            "selected_color": traj.selected_color,
            "steps": traj.steps,
            "actions": traj.actions.astype(jnp.int32),
            "old_log_probs": traj.log_probs,
            "advantages": advantages,
            "returns": returns,
            "env_id": traj.env_id,
        }

        rng, update_key = jax.random.split(rng)
        train_state, update_key, metrics = ppo_update_fn(train_state, batch, update_key, task_cache)

        rng, reset_key = jax.random.split(rng)
        state = reset_done_envs(env, next_state, reset_key, config.num_envs)

        if (update_idx + 1) % config.log_interval == 0:
            metrics_np = tree_util.tree_map(lambda x: float(x), metrics)
            returns_per_env = jnp.sum(traj.rewards, axis=0)
            summary = summarize_env_batch(env, next_state, returns_per_env)
            value_mean = float(jnp.mean(traj.values))

            log_update = update_idx + 1
            print(
                f"[update {log_update:04d}] "
                f"return={summary['return_mean']:.3f}±{summary['return_std']:.3f} "
                f"loss={metrics_np['loss']:.4f} "
                f"policy={metrics_np['policy_loss']:.4f} "
                f"value={metrics_np['value_loss']:.4f} "
                f"entropy={metrics_np['entropy']:.4f} "
                f"kl={metrics_np['approx_kl']:.4f} "
                f"clip_frac={metrics_np['clip_fraction']:.4f} "
                f"value_mean={value_mean:.4f} "
                f"done={summary['done_rate']:.3f} "
                f"success={summary['success_rate']:.3f} "
                f"steps={summary['mean_steps']:.2f} "
                f"solve_steps={summary['mean_solution_steps']:.2f} "
                f"iou={summary['mean_iou']:.3f} "
                f"valid_acc={summary['valid_accuracy']:.3f}"
            )

            if wandb_run is not None and wandb_module is not None:
                payload = {
                    "train/update": log_update,
                    "train/loss_total": metrics_np["loss"],
                    "train/loss_policy": metrics_np["policy_loss"],
                    "train/loss_value": metrics_np["value_loss"],
                    "train/entropy": metrics_np["entropy"],
                    "train/approx_kl": metrics_np["approx_kl"],
                    "train/clip_fraction": metrics_np["clip_fraction"],
                    "train/value_mean": value_mean,
                }
                payload.update({f"train/{k}": v for k, v in summary.items()})
                wandb_module.log(payload, step=log_update)

        if (update_idx + 1) % config.eval_interval == 0:
            rng, eval_metrics = evaluate_policy(
                train_state.params,
                env,
                policy_step_fn,
                task_cache_fn,
                config,
                rng,
            )
            eval_step = update_idx + 1
            print(
                f"[eval   {eval_step:04d}] "
                f"train_ret={eval_metrics['train_return']:.3f}±{eval_metrics['train_return_std']:.3f} "
                f"train_done={eval_metrics['train_done_rate']:.3f} "
                f"train_success={eval_metrics['train_success_rate']:.3f} "
                f"train_steps={eval_metrics['train_mean_steps']:.2f} "
                f"test_ret={eval_metrics['test_return']:.3f}±{eval_metrics['test_return_std']:.3f} "
                f"test_done={eval_metrics['test_done_rate']:.3f} "
                f"test_success={eval_metrics['test_success_rate']:.3f} "
                f"test_steps={eval_metrics['test_mean_steps']:.2f} "
                f"test_sol_steps={eval_metrics['test_mean_solution_steps']:.2f} "
                f"test_iou={eval_metrics['test_mean_iou']:.3f} "
                f"test_valid_acc={eval_metrics['test_valid_accuracy']:.3f}"
            )

            if wandb_run is not None and wandb_module is not None:
                wandb_module.log(
                    {f"eval/{k}": v for k, v in eval_metrics.items()
                },
                    step=eval_step,
                )
            if ckpt_dir is not None:
                maybe_save_checkpoint(ckpt_dir, update_idx + 1, train_state)
        
        # Optional: save checkpoints on independent interval if requested
        if ckpt_dir is not None and config.checkpoint_interval is not None and ((update_idx + 1) % config.checkpoint_interval == 0):
            maybe_save_checkpoint(ckpt_dir, update_idx + 1, train_state)

    if ckpt_dir is not None:
        maybe_save_checkpoint(ckpt_dir, config.total_updates, train_state)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
