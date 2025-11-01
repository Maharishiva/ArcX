from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NamedTuple, Tuple


def _select_device_from_cli(default: str = "cpu") -> str:
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
    inputs: jnp.ndarray
    targets: jnp.ndarray
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


class TaskCache(NamedTuple):
    tokens: jnp.ndarray
    pos: jnp.ndarray


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    data_dir: str = "data/training"
    num_envs: int = 8
    rollout_length: int = 32
    total_updates: int = 200
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    num_minibatches: int = 4
    num_epochs: int = 4
    max_grad_norm: float = 0.5
    eval_envs: int = 8
    eval_horizon: int = 128
    eval_interval: int = 10
    log_interval: int = 1
    checkpoint_dir: str | None = "checkpoints/ppo"


def select_cursor_logits(canvas_logits: jnp.ndarray, cursor: jnp.ndarray, grid_size: int) -> jnp.ndarray:
    """Gather the logits for the cursor locations."""
    idx = cursor[:, 0] * grid_size + cursor[:, 1]
    idx = idx[:, None, None]
    picked = jnp.take_along_axis(canvas_logits, idx, axis=1)
    return picked[:, 0, :]


def build_extra_canvas_features(
    cursor: jnp.ndarray,
    last_action: jnp.ndarray,
    selected_color: jnp.ndarray,
    steps: jnp.ndarray,
    grid_size: int,
    num_actions: int,
    max_steps: int,
) -> jnp.ndarray:
    """Construct per-cell conditioning features for the current canvas."""
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

    return jnp.concatenate([cursor_mask, last_action_feat, selected_color_feat, time_feat], axis=-1)


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


def make_task_cache_fn(model_apply):
    grid_type_ids = jnp.array([0, 1], dtype=jnp.int32)

    @jax.jit
    def cache_fn(params: Dict[str, jnp.ndarray], inputs: jnp.ndarray, targets: jnp.ndarray):
        task_grids = jnp.stack([inputs, targets], axis=1)
        ids = jnp.broadcast_to(grid_type_ids[None, :], (task_grids.shape[0], grid_type_ids.shape[0]))
        tokens, pos = model_apply(
            {"params": params},
            task_grids,
            ids,
            method=PerceiverActorCritic.prepare_task_cache,
        )
        return task_grids, ids, TaskCache(tokens, pos)

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
        task_grids: jnp.ndarray,
        grid_ids: jnp.ndarray,
        cache: TaskCache,
    ):
        extra_feats = build_extra_canvas_features(
            state.cursor,
            state.last_action,
            state.selected_color,
            state.steps,
            grid_size,
            num_actions,
            max_steps,
        )
        outputs = model_apply(
            {"params": params},
            task_grids,
            grid_ids,
            state.canvas,
            extra_feats,
            cached_task_tokens=cache.tokens,
            cached_task_pos=cache.pos,
            deterministic=True,
        )
        cursor_logits = select_cursor_logits(outputs["logits"], state.cursor, grid_size)
        log_probs_all = jax.nn.log_softmax(cursor_logits, axis=-1)
        return log_probs_all, outputs["value"]

    def policy_step(
        params: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
        state: ARCEnvState,
        task_grids: jnp.ndarray,
        grid_ids: jnp.ndarray,
        cache: TaskCache,
    ):
        log_probs_all, values = policy_forward(params, state, task_grids, grid_ids, cache)
        actions = jax.random.categorical(rng_key, log_probs_all, axis=-1).astype(jnp.int32)
        idx = jnp.arange(actions.shape[0])
        log_probs = log_probs_all[idx, actions]
        return actions, log_probs, values

    @jax.jit
    def policy_step_greedy(
        params: Dict[str, jnp.ndarray],
        state: ARCEnvState,
        task_grids: jnp.ndarray,
        grid_ids: jnp.ndarray,
        cache: TaskCache,
    ):
        log_probs_all, values = policy_forward(params, state, task_grids, grid_ids, cache)
        actions = jnp.argmax(log_probs_all, axis=-1).astype(jnp.int32)
        idx = jnp.arange(actions.shape[0])
        log_probs = log_probs_all[idx, actions]
        return actions, log_probs, values

    @jax.jit
    def rollout(
        params: Dict[str, jnp.ndarray],
        state: ARCEnvState,
        rng: jax.Array,
        task_grids: jnp.ndarray,
        grid_ids: jnp.ndarray,
        cache: TaskCache,
    ):
        def step(carry, _):
            key, env_state = carry
            key, step_key = jax.random.split(key)
            actions, log_probs, values = policy_step(params, step_key, env_state, task_grids, grid_ids, cache)
            next_state, rewards, dones = env.env_step_batch(env_state, actions)
            transition = (
                env_state.inp,
                env_state.target,
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
            )
            return (key, next_state), transition

        (final_key, final_state), trajectory = jax.lax.scan(
            step,
            (rng, state),
            None,
            length=config.rollout_length,
        )
        return final_state, RolloutBatch(*trajectory), final_key

    return rollout, policy_step_greedy


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
        task_grids: jnp.ndarray,
        grid_ids: jnp.ndarray,
        cache: TaskCache,
    ) -> jnp.ndarray:
        extra_feats = build_extra_canvas_features(
            state.cursor,
            state.last_action,
            state.selected_color,
            state.steps,
            grid_size,
            num_actions,
            max_steps,
        )
        outputs = model_apply(
            {"params": params},
            task_grids,
            grid_ids,
            state.canvas,
            extra_feats,
            cached_task_tokens=cache.tokens,
            cached_task_pos=cache.pos,
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
    grid_type_ids = jnp.array([0, 1], dtype=jnp.int32)
    batch_size = config.num_envs * config.rollout_length
    minibatch_size = batch_size // config.num_minibatches

    def loss_fn(params, batch):
        task_grids = jnp.stack([batch["inputs"], batch["targets"]], axis=1)
        grid_ids = jnp.broadcast_to(grid_type_ids[None, :], (task_grids.shape[0], grid_type_ids.shape[0]))
        extra_feats = build_extra_canvas_features(
            batch["cursor"],
            batch["last_action"],
            batch["selected_color"],
            batch["steps"],
            grid_size,
            num_actions,
            max_steps,
        )
        outputs = model_apply({"params": params}, task_grids, grid_ids, batch["canvas"], extra_feats, deterministic=False)
        cursor_logits = select_cursor_logits(outputs["logits"], batch["cursor"], grid_size)
        log_probs_all = jax.nn.log_softmax(cursor_logits, axis=-1)
        idx = jnp.arange(cursor_logits.shape[0])
        new_log_probs = log_probs_all[idx, batch["actions"]]

        ratios = jnp.exp(new_log_probs - batch["old_log_probs"])
        clipped = jnp.clip(ratios, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        policy_loss = -jnp.mean(jnp.minimum(ratios * batch["advantages"], clipped * batch["advantages"]))

        values = outputs["value"]
        value_loss = 0.5 * jnp.mean((values - batch["returns"]) ** 2)

        entropy = -jnp.mean(jnp.sum(jnp.exp(log_probs_all) * log_probs_all, axis=-1))
        approx_kl = jnp.mean(batch["old_log_probs"] - new_log_probs)
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
    def update(train_state: TrainState, batch: Dict[str, jnp.ndarray], rng: jax.Array):
        def epoch_step(carry, _):
            state, key = carry
            key, perm_key = jax.random.split(key)
            permutation = jax.random.permutation(perm_key, batch_size)
            shuffled = {k: jnp.take(v, permutation, axis=0) for k, v in batch.items()}
            reshaped = {k: v.reshape((config.num_minibatches, minibatch_size) + v.shape[1:]) for k, v in shuffled.items()}

            def minibatch_step(state, minibatch):
                (loss_val, metrics), grads = grad_fn(state.params, minibatch)
                state = state.apply_gradients(grads=grads)
                return state, metrics

            state, mb_metrics = jax.lax.scan(minibatch_step, state, reshaped)
            metrics_avg = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), mb_metrics)
            return (state, key), metrics_avg

        (new_state, new_key), epoch_metrics = jax.lax.scan(epoch_step, (train_state, rng), jnp.arange(config.num_epochs))
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
        selected_color=_select(reset_state.selected_color, state.selected_color),
        last_action=_select(reset_state.last_action, state.last_action),
    )


def evaluate_policy(
    params: Dict[str, jnp.ndarray],
    env,
    policy_step_greedy,
    task_cache_fn,
    config: PPOConfig,
    rng: jax.Array,
) -> Tuple[jax.Array, Dict[str, float]]:
    def run_split(rng_key: jax.Array, train_flag: bool):
        rng_key, reset_key = jax.random.split(rng_key)
        state = env.env_reset_batch(reset_key, train=train_flag, batch_size=config.eval_envs)
        task_grids, grid_ids, cache = task_cache_fn(params, state.inp, state.target)
        returns = jnp.zeros((config.eval_envs,), dtype=jnp.float32)

        for _ in range(config.eval_horizon):
            actions, _, _ = policy_step_greedy(params, state, task_grids, grid_ids, cache)
            state, rewards, dones = env.env_step_batch(state, actions)
            returns = returns + rewards
            if bool(jnp.all(dones)):
                break

        metrics = {
            "return": float(jnp.mean(returns)),
            "max_return": float(jnp.max(returns)),
            "done_rate": float(jnp.mean(state.done.astype(jnp.float32))),
            "mean_steps": float(jnp.mean(state.steps.astype(jnp.float32))),
        }
        return rng_key, metrics

    rng, train_metrics = run_split(rng, train_flag=True)
    rng, test_metrics = run_split(rng, train_flag=False)

    eval_metrics = {}
    for key, value in train_metrics.items():
        eval_metrics[f"train_{key}"] = value
    for key, value in test_metrics.items():
        eval_metrics[f"test_{key}"] = value

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
    parser.add_argument("--checkpoint-dir", type=str, default=None)
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
    overrides = {}
    for key, value in vars(args).items():
        if key in field_names and value is not None:
            overrides[key] = value

    if overrides:
        config = dataclasses.replace(config, **overrides)

    if config.max_grad_norm is not None and config.max_grad_norm < 0:
        config = dataclasses.replace(config, max_grad_norm=None)

    active_device = args.device or _requested_device or "cpu"
    return config, active_device


def main():
    config, device_choice = parse_args()
    assert (config.num_envs * config.rollout_length) % config.num_minibatches == 0, "Minibatch size must divide rollout size"

    cpu_devices = jax.devices("cpu")
    load_ctx = jax.default_device(cpu_devices[0]) if cpu_devices else nullcontext()
    with load_ctx:
        env = build_env_from_dir(Path(config.data_dir))
    grid_size = env.GRID_SIZE
    num_actions = env.NUM_ACTIONS
    max_steps = env.max_steps

    model = PerceiverActorCritic()

    rng = jax.random.PRNGKey(config.seed)
    rng, reset_key = jax.random.split(rng)
    state = env.env_reset_batch(reset_key, train=True, batch_size=config.num_envs)

    task_grids = jnp.stack([state.inp, state.target], axis=1)
    grid_ids = jnp.broadcast_to(jnp.array([0, 1], dtype=jnp.int32)[None, :], (config.num_envs, 2))

    init_extra_feats = build_extra_canvas_features(
        state.cursor,
        state.last_action,
        state.selected_color,
        state.steps,
        grid_size,
        num_actions,
        max_steps,
    )

    rng, init_key = jax.random.split(rng)
    params = model.init(init_key, task_grids, grid_ids, state.canvas, init_extra_feats)

    if config.max_grad_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    else:
        tx = optax.adam(config.learning_rate)
    train_state = TrainState.create(apply_fn=model.apply, params=params["params"], tx=tx)

    rollout_fn, policy_step_greedy = make_rollout_fn(model.apply, env, config, grid_size, num_actions, max_steps)
    value_fn = make_value_fn(model.apply, grid_size, num_actions, max_steps)
    ppo_update_fn = make_ppo_update_fn(model.apply, config, grid_size, num_actions, max_steps)
    task_cache_fn = make_task_cache_fn(model.apply)

    param_count = sum(int(x.size) for x in tree_util.tree_leaves(train_state.params))
    print(
        f"Initialized PerceiverActorCritic with {param_count / 1e6:.2f}M trainable parameters "
        f"on device preset '{device_choice}'."
    )

    ckpt_dir = Path(config.checkpoint_dir).resolve() if config.checkpoint_dir is not None else None

    for update_idx in range(config.total_updates):
        rng, rollout_key = jax.random.split(rng)
        task_grids, grid_ids, task_cache = task_cache_fn(train_state.params, state.inp, state.target)
        next_state, traj, rollout_key = rollout_fn(
            train_state.params,
            state,
            rollout_key,
            task_grids,
            grid_ids,
            task_cache,
        )

        next_task_grids, next_grid_ids, next_task_cache = task_cache_fn(
            train_state.params, next_state.inp, next_state.target
        )
        last_value = value_fn(
            train_state.params,
            next_state,
            next_task_grids,
            next_grid_ids,
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

        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - jnp.mean(adv_flat)) / (jnp.std(adv_flat) + 1e-8)

        batch = {
            "inputs": flatten_batch(traj.inputs),
            "targets": flatten_batch(traj.targets),
            "canvas": flatten_batch(traj.canvas),
            "cursor": flatten_batch(traj.cursor),
            "last_action": flatten_batch(traj.last_action),
            "selected_color": flatten_batch(traj.selected_color),
            "steps": flatten_batch(traj.steps),
            "actions": flatten_batch(traj.actions).astype(jnp.int32),
            "old_log_probs": flatten_batch(traj.log_probs),
            "advantages": adv_flat,
            "returns": returns.reshape(-1),
        }

        rng, update_key = jax.random.split(rng)
        train_state, update_key, metrics = ppo_update_fn(train_state, batch, update_key)

        rng, reset_key = jax.random.split(rng)
        state = reset_done_envs(env, next_state, reset_key, config.num_envs)

        if (update_idx + 1) % config.log_interval == 0:
            metrics_np = tree_util.tree_map(lambda x: float(x), metrics)
            rollout_return = float(jnp.mean(jnp.sum(traj.rewards, axis=0)))
            value_mean = float(jnp.mean(traj.values))
            print(
                f"[update {update_idx + 1:04d}] "
                f"return={rollout_return:.3f} "
                f"loss={metrics_np['loss']:.4f} "
                f"policy={metrics_np['policy_loss']:.4f} "
                f"value={metrics_np['value_loss']:.4f} "
                f"entropy={metrics_np['entropy']:.4f} "
                f"kl={metrics_np['approx_kl']:.4f} "
                f"clip_frac={metrics_np['clip_fraction']:.4f} "
                f"value_mean={value_mean:.4f}"
            )

        if (update_idx + 1) % config.eval_interval == 0:
            rng, eval_metrics = evaluate_policy(
                train_state.params,
                env,
                policy_step_greedy,
                task_cache_fn,
                config,
                rng,
            )
            print(
                f"[eval   {update_idx + 1:04d}] "
                f"train_return={eval_metrics['train_return']:.3f} "
                f"train_done={eval_metrics['train_done_rate']:.3f} "
                f"test_return={eval_metrics['test_return']:.3f} "
                f"test_done={eval_metrics['test_done_rate']:.3f} "
                f"test_steps={eval_metrics['test_mean_steps']:.2f}"
            )
            if ckpt_dir is not None:
                maybe_save_checkpoint(ckpt_dir, update_idx + 1, train_state)

    if ckpt_dir is not None:
        maybe_save_checkpoint(ckpt_dir, config.total_updates, train_state)


if __name__ == "__main__":
    main()
