"""Integration test for ARC env and Perceiver actor-critic."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from env.env import ARCEnv
from nets.PerceiverActorCritic import PerceiverActorCritic


def _build_small_env(max_steps: int = ARCEnv.DEFAULT_MAX_STEPS) -> ARCEnv:
    """Create a minimal inline ARC task to keep the test fast and portable."""
    sample_json = """{
        "train": [
            {"input": [[1,2,3],[4,5,6],[7,8,9]], "output": [[1,1,1],[2,2,2],[3,3,3]]},
            {"input": [[0,0,0],[1,1,1],[2,2,2]], "output": [[9,9,9],[8,8,8],[7,7,7]]}
        ],
        "test": [
            {"input": [[5,5,5],[6,6,6],[7,7,7]], "output": [[1,2,3],[4,5,6],[7,8,9]]}
        ]
    }"""
    return ARCEnv.from_json(sample_json, max_steps=max_steps)


def _make_model_inputs(env: ARCEnv, state):
    """Prepare model inputs (batch=1) from env state."""
    problem_idx = jnp.asarray(state.problem_idx)[None]
    task_grids, grid_type_ids, task_mask = env.build_context_batch(problem_idx)
    canvas_grid = state.canvas[None, ...]  # (1, 30, 30)
    return task_grids, grid_type_ids, canvas_grid, task_mask


class EnvModelIntegrationTest(unittest.TestCase):
    def test_context_batch_includes_all_train_pairs(self):
        env = _build_small_env()
        state = env.env_reset(jax.random.PRNGKey(42), train=True)
        problem_idx = jnp.asarray(state.problem_idx)[None]
        task_grids, grid_ids, mask = env.build_context_batch(problem_idx)

        expected_slots = env.max_demo_pairs * 2
        self.assertEqual(task_grids.shape, (1, expected_slots, env.GRID_SIZE, env.GRID_SIZE))
        self.assertEqual(grid_ids.shape, (1, expected_slots))

        pair_ids = jnp.arange(env.max_demo_pairs, dtype=jnp.int32)
        expected_grid_ids = jnp.stack([pair_ids, pair_ids + env.max_demo_pairs], axis=1).reshape(-1)
        self.assertTrue(jnp.array_equal(grid_ids[0], expected_grid_ids))

        # First two training pairs should populate the first four slots (input/output pairs)
        self.assertTrue(jnp.array_equal(task_grids[0, 0], env.train_input[0]))
        self.assertTrue(jnp.array_equal(task_grids[0, 1], env.train_output[0]))
        self.assertTrue(jnp.array_equal(task_grids[0, 2], env.train_input[1]))
        self.assertTrue(jnp.array_equal(task_grids[0, 3], env.train_output[1]))

        padded = task_grids[0, 4:]
        self.assertTrue(jnp.all(padded == env.EMPTY_CELL))

        mask_grid = mask.reshape(1, expected_slots, env.GRID_SIZE * env.GRID_SIZE)
        self.assertTrue(bool(jnp.all(mask_grid[0, :4])))
        self.assertTrue(bool(jnp.all(mask_grid[0, 4:] == 0)))

    def test_env_and_model_rollout(self):
        env = _build_small_env(max_steps=100)
        rng = jax.random.PRNGKey(0)
        state = env.env_reset(rng, train=True)

        model = PerceiverActorCritic()
        dummy_task, dummy_ids, dummy_canvas, dummy_mask = _make_model_inputs(env, state)
        params = model.init(jax.random.PRNGKey(1), dummy_task, dummy_ids, dummy_canvas, task_mask=dummy_mask)

        @jax.jit
        def policy_fn(params, rng, task_grids, grid_ids, canvas, task_mask):
            outputs = model.apply(params, task_grids, grid_ids, canvas, task_mask=task_mask, deterministic=True)
            pooled_logits = outputs["logits"].mean(axis=1)  # (B, policy_dim)
            actions = jax.random.categorical(rng, pooled_logits, axis=-1)
            return actions.astype(jnp.int32), outputs["value"]

        steps_taken = 0
        total_reward = 0.0
        for _ in range(env.max_steps):
            task_grids, grid_ids, canvas, task_mask = _make_model_inputs(env, state)
            state_rng, action_rng = jax.random.split(state.rng)
            state = replace(state, rng=state_rng)
            actions, values = policy_fn(params, action_rng, task_grids, grid_ids, canvas, task_mask)
            action = int(actions[0])
            state, reward, done = env.env_step(state, jnp.array(action, dtype=jnp.int32))
            steps_taken += 1
            total_reward += float(reward)
            self.assertFalse(jnp.isnan(values).any())
            self.assertEqual(reward.dtype, jnp.float32)
            if bool(done):
                break

        # Verify episode terminated properly
        self.assertTrue(bool(state.done), "Episode should be done after loop")
        self.assertEqual(int(state.steps), steps_taken, "Steps count should match")
        # Either hit max_steps or terminated early via SEND
        self.assertTrue(steps_taken <= env.max_steps, "Should not exceed max_steps")


if __name__ == "__main__":
    unittest.main()
