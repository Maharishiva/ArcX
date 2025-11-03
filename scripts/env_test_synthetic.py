#!/usr/bin/env python
"""Run a fixed action sequence on a synthetic ARC task and export a GIF."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import imageio.v3 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.env import ARCEnv

SCALE = 8
FRAME_REPEAT = 4

# ARC colour palette (index 10 is used for EMPTY_CELL padding).
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
    mapped = np.array(grid, dtype=np.int32)
    mapped = np.where(mapped < 0, 10, mapped)
    mapped = np.clip(mapped, 0, 10)
    return PALETTE[mapped]


def compute_score_components(state) -> dict:
    canvas = np.array(state.canvas)
    target = np.array(state.target)
    valid_mask = np.array(state.valid_mask, dtype=bool)

    painted_mask = canvas != ARCEnv.EMPTY_CELL
    intersection = np.logical_and(painted_mask, valid_mask).sum()
    union = np.logical_or(painted_mask, valid_mask).sum()
    iou = float(intersection / union) if union > 0 else 1.0

    valid_cells = valid_mask.sum()
    if valid_cells > 0:
        valid_acc = float((canvas == target)[valid_mask].mean())
    else:
        valid_acc = 1.0

    full_match = bool(np.array_equal(canvas, target))
    full_component = 1.0 if full_match else 0.0
    score = 0.2 * iou + valid_acc + full_component

    return {
        "score": score,
        "iou": iou,
        "valid_acc": valid_acc,
        "full_match": full_match,
        "full_component": full_component,
        "iou_component": 0.2 * iou,
    }


def render_info_panel(height: int, lines: List[str]) -> np.ndarray:
    width = 280
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 4
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0), font=font)
        y += line_height
    return np.array(image, dtype=np.uint8)


def build_frame(
    state,
    task_input: np.ndarray,
    task_target: np.ndarray,
    grid_size: int,
    info_lines: List[str],
) -> np.ndarray:
    canvas_img = grid_to_rgb(state.canvas)
    input_img = grid_to_rgb(task_input)
    target_img = grid_to_rgb(task_target)

    selected_idx = int(np.clip(state.selected_color, 0, len(PALETTE) - 1))
    selected_color = PALETTE[selected_idx][None, None, :]
    color_strip = np.tile(selected_color, (grid_size, 4, 1))

    spacer = np.full((grid_size, 2, 3), 255, dtype=np.uint8)
    base = np.concatenate(
        [input_img, spacer, canvas_img, spacer, target_img, spacer, color_strip],
        axis=1,
    )

    scaled = np.repeat(np.repeat(base, SCALE, axis=0), SCALE, axis=1)

    canvas_offset = input_img.shape[1] + spacer.shape[1]
    cell_y = int(np.clip(state.cursor[0], 0, grid_size - 1))
    cell_x = int(np.clip(state.cursor[1], 0, grid_size - 1))
    y0 = cell_y * SCALE
    y1 = (cell_y + 1) * SCALE
    x0 = (canvas_offset + cell_x) * SCALE
    x1 = (canvas_offset + cell_x + 1) * SCALE
    border = np.array([255, 0, 0], dtype=np.uint8)
    scaled[y0:y1, x0 : x0 + 1] = border
    scaled[y0:y1, x1 - 1 : x1] = border
    scaled[y0 : y0 + 1, x0:x1] = border
    scaled[y1 - 1 : y1, x0:x1] = border

    info_panel = render_info_panel(scaled.shape[0], info_lines)

    return np.concatenate([scaled, info_panel], axis=1)


def make_env() -> ARCEnv:
    sample_json = """{
        "train": [],
        "test": [
            {
                "input": [[0,0,0],[0,1,2],[0,3,4]],
                "output": [[1,2,-1],[3,4,-1],[-1,-1,-1]]
            }
        ]
    }"""
    return ARCEnv.from_json(sample_json, max_steps=64)


def run_sequence(env: ARCEnv, gif_path: Path, fps: float = 0.1) -> None:
    state = env.env_reset(jax.random.PRNGKey(0), train=False)
    grid_size = env.GRID_SIZE
    task_input = np.array(state.inp)
    task_target = np.array(state.target)

    metrics = compute_score_components(state)
    total_reward = 0.0
    initial_info = [
        "Step: 0",
        "Action: START",
        "Reward: +0.000",
        "Total Reward: +0.000",
        f"Score: {metrics['score']:.3f}",
        f"- 0.2 * IoU: {metrics['iou_component']:.3f}",
        f"- Valid Acc: {metrics['valid_acc']:.3f}",
        f"- Full Match: {metrics['full_component']:.3f}",
        f"IoU: {metrics['iou']:.3f}",
        f"Valid Acc: {metrics['valid_acc']:.3f}",
        f"Full Match: {'Yes' if metrics['full_match'] else 'No'}",
        "Done: No",
    ]

    frames: List[np.ndarray] = [build_frame(state, task_input, task_target, grid_size, initial_info)]
    rewards: List[Tuple[str, float, bool]] = [("START", 0.0, False)]

    choose = ARCEnv.ACT_CHOOSE_COLOR_START
    actions = [
        (ARCEnv.ACT_COPY, "COPY"),
        (ARCEnv.ACT_RIGHT, "RIGHT"),
        (ARCEnv.ACT_DOWN, "DOWN"),
        (choose + 4, "CHOOSE_COLOR_4"),
        (ARCEnv.ACT_PAINT, "PAINT"),
        (ARCEnv.ACT_MOVE_TO_ORIGIN, "MOVE_TO_ORIGIN"),
        (ARCEnv.ACT_RIGHT, "RIGHT"),
        # (ARCEnv.ACT_RIGHT, "RIGHT"),
        (ARCEnv.ACT_DOWN, "DOWN"),
        # (ARCEnv.ACT_DOWN, "DOWN"),
        (ARCEnv.ACT_CROP, "CROP"),
        (ARCEnv.ACT_LEFT, "LEFT"),
        (ARCEnv.ACT_LEFT, "LEFT"),
        (ARCEnv.ACT_UP, "UP"),
        (ARCEnv.ACT_UP, "UP"),
        (choose + 1, "CHOOSE_COLOR_1"),
        (ARCEnv.ACT_PAINT, "PAINT"),
        (ARCEnv.ACT_SEND, "SEND"),
    ]

    for step_idx, (action_id, label) in enumerate(actions, start=1):
        next_state, reward, done = env.env_step(state, jnp.array(action_id, dtype=jnp.int32))
        reward_val = float(reward)
        done_flag = bool(done)
        total_reward += reward_val

        metrics = compute_score_components(next_state)
        info_lines = [
            f"Step: {step_idx}",
            f"Action: {label}",
            f"Reward: {reward_val:+.3f}",
            f"Total Reward: {total_reward:+.3f}",
            f"Score: {metrics['score']:.3f}",
            f"- 0.2 * IoU: {metrics['iou_component']:.3f}",
            f"- Valid Acc: {metrics['valid_acc']:.3f}",
            f"- Full Match: {metrics['full_component']:.3f}",
            f"IoU: {metrics['iou']:.3f}",
            f"Valid Acc: {metrics['valid_acc']:.3f}",
            f"Full Match: {'Yes' if metrics['full_match'] else 'No'}",
            f"Done: {'Yes' if done_flag else 'No'}",
        ]

        rewards.append((label, reward_val, done_flag))
        frames.append(build_frame(next_state, task_input, task_target, grid_size, info_lines))
        state = next_state
        if done_flag:
            break

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    raw_frames = np.stack(frames)
    slowed = np.repeat(raw_frames, FRAME_REPEAT, axis=0)
    frame_duration = 1.0 / (fps * FRAME_REPEAT) if fps > 0 else 0.1
    imageio.imwrite(gif_path, slowed, format="GIF", duration=frame_duration, loop=0)

    print(f"Saved GIF to {gif_path.resolve()}")
    print("\nStep-by-step rewards:")
    total = 0.0
    for idx, (label, reward, done_flag) in enumerate(rewards[1:], start=1):
        total += reward
        status = " (done)" if done_flag else ""
        print(f"{idx:02d}. {label:<16} reward={reward:+.3f}{status}")
    print(f"\nTotal reward: {total:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed action sequence on a synthetic ARC task and export a GIF.")
    parser.add_argument("--fps", type=float, default=0.1, help="Frames per second for the output GIF (default: 0.1)")
    args = parser.parse_args()

    env = make_env()
    gif_path = Path("artifacts") / "synthetic_env_sequence.gif"
    run_sequence(env, gif_path, fps=args.fps)


if __name__ == "__main__":
    main()
