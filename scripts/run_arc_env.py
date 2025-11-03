"""Demo script for the ARC environment with new actions."""

import os
import argparse
from pathlib import Path

# Force CPU backend by default to avoid experimental Metal issues on macOS
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

from env import ARCEnv
from env.data_loader import build_env_from_dir


def demo_basic_actions(env: ARCEnv, render: bool = False):
    """Demo basic painting and movement actions."""
    print("\n=== Demo: Basic Actions ===")
    rng = jax.random.PRNGKey(0)
    state = env.env_reset(rng, train=True)
    
    # Paint some colors, move cursor, copy from input
    actions = [4, 5, 3, 3, 14, 0, 0, 12]
    action_names = ["paint 0", "paint 1", "right", "right", "copy", "up", "up", "paint 8"]
    
    for a, name in zip(actions, action_names):
        state, reward, done = env.env_step(state, jnp.array(a))
        print(f"Action {a:2d} ({name:12s}): reward={float(reward):6.1f}, done={bool(done)}, cursor={state.cursor}")
        if render:
            env.env_render(state, mode="canvasOnly")
        if bool(done):
            break


def demo_erase_action(env: ARCEnv, render: bool = False):
    """Demo the erase action (paint -1)."""
    print("\n=== Demo: Erase Action (15) ===")
    rng = jax.random.PRNGKey(1)
    state = env.env_reset(rng, train=True)
    
    # Paint, then erase
    actions = [4, 1, 1, 5, 0, 0, 15]  # paint 0, down, down, paint 1, up, up, erase
    action_names = ["paint 0", "down", "down", "paint 1", "up", "up", "erase"]
    
    for a, name in zip(actions, action_names):
        state, reward, done = env.env_step(state, jnp.array(a))
        print(f"Action {a:2d} ({name:12s}): reward={float(reward):6.1f}, cursor={state.cursor}")
        if render and name in ["paint 1", "erase"]:
            env.env_render(state, mode="canvasOnly")


def demo_crop_action(env: ARCEnv, render: bool = False):
    """Demo the crop action (fill right and down with -1)."""
    print("\n=== Demo: Crop Action (16) ===")
    rng = jax.random.PRNGKey(2)
    state = env.env_reset(rng, train=True)
    
    # Paint a pattern, move cursor, then crop
    actions = [4, 3, 5, 3, 6, 1, 2, 2, 7, 1, 16]
    action_names = ["paint 0", "right", "paint 1", "right", "paint 2", 
                    "down", "left", "left", "paint 3", "down", "crop"]
    
    for a, name in zip(actions, action_names):
        state, reward, done = env.env_step(state, jnp.array(a))
        print(f"Action {a:2d} ({name:12s}): cursor={state.cursor}")
        if render and name in ["paint 3", "crop"]:
            print(f"  Showing canvas after: {name}")
            env.env_render(state, mode="canvasOnly")


def demo_move_to_origin(env: ARCEnv, render: bool = False):
    """Demo the move-to-origin action (shift grid)."""
    print("\n=== Demo: Move-to-Origin Action (17) ===")
    rng = jax.random.PRNGKey(3)
    state = env.env_reset(rng, train=True)
    
    # Paint in an offset position, then shift to origin
    # Move cursor to (5, 7), paint, then shift
    moves_to_5_7 = [1] * 5 + [3] * 7  # down 5, right 7
    actions = moves_to_5_7 + [4, 3, 5, 1, 4, 17]  # paint some, then shift
    
    for i, a in enumerate(actions):
        state, reward, done = env.env_step(state, jnp.array(a))
        if i == len(moves_to_5_7) - 1:
            print(f"After moving to (5,7): cursor={state.cursor}")
        elif i == len(moves_to_5_7) + 4:
            print(f"After painting at offset: cursor={state.cursor}")
            if render:
                env.env_render(state, mode="canvasOnly")
        elif a == 17:
            print(f"After shift-to-origin: cursor={state.cursor}")
            if render:
                env.env_render(state, mode="canvasOnly")


def main():
    parser = argparse.ArgumentParser(description="ARC Environment Demo")
    parser.add_argument("--data_dir", type=str, default="", 
                       help="Directory containing ARC JSON files")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--render", action="store_true",
                       help="Render canvas during demos")
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense"],
        help="Reward schedule: sparse (terminal only) or dense (per-step).",
    )
    parser.add_argument("--demo", type=str, default="all",
                       choices=["all", "basic", "erase", "crop", "shift"],
                       help="Which demo to run")
    args = parser.parse_args()

    # Build environment
    if args.data_dir:
        env = build_env_from_dir(
            Path(args.data_dir),
            max_steps=args.max_steps,
            reward_mode=args.reward_mode,
            max_demo_pairs=5,
        )
        print(f"Loaded environment from {args.data_dir}")
        print(f"  Train examples: {env.train_input.shape[0]}")
        print(f"  Test examples: {env.test_input.shape[0]}")
    else:
        # Minimal inline example if no dataset provided
        sample_data_json = (
            "{"
            "\"train\": [{\"input\": [[6,6,0],[6,0,0],[0,6,6]], "
            "\"output\": [[6,6,0,6,6,0,0,0,0],[6,0,0,6,0,0,0,0,0],[0,6,6,0,6,6,0,0,0]]}],"
            "\"test\": []}"
        )
        env = ARCEnv.from_json(sample_data_json, max_steps=args.max_steps)
        print("Using built-in sample task")

    print(f"\nEnvironment Info:")
    print(f"  Grid size: {env.GRID_SIZE}x{env.GRID_SIZE}")
    print(f"  Number of actions: {env.NUM_ACTIONS}")
    print(f"  Max steps per episode: {env.max_steps}")
    print(f"  Actions: 0-3 move, 4-13 paint colors 0-9, 14 copy, 15 erase, 16 crop, 17 shift-to-origin")

    # Run selected demos
    if args.demo in ["all", "basic"]:
        demo_basic_actions(env, render=args.render)
    
    if args.demo in ["all", "erase"]:
        demo_erase_action(env, render=args.render)
    
    if args.demo in ["all", "crop"]:
        demo_crop_action(env, render=args.render)
    
    if args.demo in ["all", "shift"]:
        demo_move_to_origin(env, render=args.render)

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
