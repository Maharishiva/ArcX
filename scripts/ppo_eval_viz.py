from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List

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

import imageio.v3 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from env.data_loader import build_env_from_dir
from nets.PerceiverActorCritic import PerceiverActorCritic
from scripts.ppo_train import (
    TaskCache,
    build_extra_canvas_features,
    make_task_cache_fn,
    select_cursor_logits,
)

PALETTE = np.array(
    [
        (0, 0, 0),
        (0, 116, 217),
        (255, 65, 54),
        (46, 204, 64),
        (255, 220, 0),
        (170, 170, 170),
        (240, 18, 190),
        (255, 133, 27),
        (127, 219, 255),
        (135, 12, 37),
        (255, 255, 255),
    ],
    dtype=np.uint8,
)


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Convert a 30x30 int grid into an RGB image using the ARC palette."""
    mapped = np.array(grid, dtype=np.int32)
    mapped = np.where(mapped < 0, 10, mapped)
    mapped = np.clip(mapped, 0, 10)
    return PALETTE[mapped]


def highlight_cursor(image: np.ndarray, cursor: np.ndarray) -> np.ndarray:
    """Overlay a small crosshair at the cursor position."""
    y, x = int(cursor[0]), int(cursor[1])
    h, w, _ = image.shape
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)
    img = image.copy()
    img[y, max(0, x - 1) : min(w, x + 2)] = np.array([255, 0, 0], dtype=np.uint8)
    img[max(0, y - 1) : min(h, y + 2), x] = np.array([255, 0, 0], dtype=np.uint8)
    return img


def build_frame(state, task_inputs: np.ndarray, task_targets: np.ndarray, grid_size: int) -> np.ndarray:
    """Create a side-by-side frame of input, canvas, and target grids."""
    inp_img = grid_to_rgb(task_inputs)
    target_img = grid_to_rgb(task_targets)
    canvas_img = grid_to_rgb(state.canvas[0])
    canvas_img = highlight_cursor(canvas_img, np.array(state.cursor[0]))

    selected_idx = int(np.clip(state.selected_color[0], 0, len(PALETTE) - 1))
    selected_color = PALETTE[selected_idx][None, None, :]
    color_strip = np.tile(selected_color, (grid_size, 4, 1))

    spacer = np.full((grid_size, 2, 3), 255, dtype=np.uint8)
    base_row = np.concatenate([inp_img, spacer, canvas_img, spacer, target_img, spacer, color_strip], axis=1)
    return base_row


def make_greedy_policy(model, grid_size: int, num_actions: int, max_steps: int):
    @jax.jit
    def policy_fn(
        params: dict,
        state,
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
        outputs = model.apply(
            {"params": params},
            task_grids,
            grid_ids,
            state.canvas,
            extra_feats,
            cached_task_tokens=cache.tokens,
            cached_task_pos=cache.pos,
            deterministic=True,
        )
        logits = select_cursor_logits(outputs["logits"], state.cursor, grid_size)
        actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return actions, outputs["value"]

    return policy_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Render PPO policy rollouts as GIFs on ARC evaluation tasks.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory or file for the PPO model.")
    parser.add_argument("--data-dir", type=str, default="data/training", help="Path to ARC data directory.")
    parser.add_argument("--output-dir", type=str, default="artifacts/ppo_eval", help="Directory to store output GIFs.")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of evaluation tasks to render.")
    parser.add_argument("--rollout-horizon", type=int, default=64, help="Maximum steps per rollout.")
    parser.add_argument("--fps", type=int, default=4, help="Playback FPS for the generated GIF.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation sampling.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "tpu", "auto"], default=None, help="Device override for JAX runtime.")
    return parser.parse_args()


def load_params(model: PerceiverActorCritic, checkpoint: Path):
    restored = checkpoints.restore_checkpoint(checkpoint.resolve(), target=None)
    if hasattr(restored, "params"):
        return restored.params
    if isinstance(restored, dict) and "params" in restored:
        return restored["params"]
    raise ValueError(f"Unable to load parameters from checkpoint: {checkpoint}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_device = args.device or _requested_device or "cpu"
    print(f"Using device preset '{active_device}' for evaluation.")

    env = build_env_from_dir(Path(args.data_dir))
    grid_size = env.GRID_SIZE
    num_actions = env.NUM_ACTIONS
    max_steps = env.max_steps

    model = PerceiverActorCritic()
    params = load_params(model, Path(args.checkpoint))

    task_cache_fn = make_task_cache_fn(model.apply)
    greedy_policy = make_greedy_policy(model, grid_size, num_actions, max_steps)

    rng = jax.random.PRNGKey(args.seed)

    for episode in range(args.num_episodes):
        rng, reset_key = jax.random.split(rng)
        state = env.env_reset_batch(reset_key, train=False, batch_size=1)
        task_grids, grid_ids, cache = task_cache_fn(params, state.inp, state.target)

        frames: List[np.ndarray] = []
        total_reward = 0.0

        inp_np = np.array(state.inp[0])
        tgt_np = np.array(state.target[0])

        for step in range(args.rollout_horizon):
            frames.append(build_frame(state, inp_np, tgt_np, grid_size))
            actions, values = greedy_policy(params, state, task_grids, grid_ids, cache)
            state, reward, done = env.env_step_batch(state, actions)
            total_reward += float(reward[0])
            if bool(done[0]):
                frames.append(build_frame(state, inp_np, tgt_np, grid_size))
                break

        gif_path = output_dir / f"episode_{episode:03d}.gif"
        imageio.imwrite(gif_path, frames, loop=0, duration=1.0 / max(args.fps, 1))
        print(
            f"Episode {episode:02d}: return={total_reward:.3f} "
            f"steps={int(state.steps[0])} "
            f"done={bool(state.done[0])} -> saved {gif_path}"
        )


if __name__ == "__main__":
    main()
