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

from env.data_loader import load_arc_json_file
from env.env import ARCEnv
from nets.PerceiverActorCritic import PerceiverActorCritic


def _build_small_env(max_steps: int = ARCEnv.DEFAULT_MAX_STEPS) -> ARCEnv:
    """Load a single ARC task to keep the test fast."""
    first_task = sorted(Path("data/training").glob("*.json"))[0]
    train_in, train_out, test_in, test_out = load_arc_json_file(first_task)
    return ARCEnv(train_in, train_out, test_in, test_out, max_steps=max_steps)


def _make_model_inputs(state):
    """Prepare model inputs (batch=1) from env state."""
    task_grids = jnp.stack([state.inp, state.target])[None, ...]  # (1, 2, 30, 30)
    grid_type_ids = jnp.array([[0, 1]], dtype=jnp.int32)
    canvas_grid = state.canvas[None, ...]  # (1, 30, 30)
    return task_grids, grid_type_ids, canvas_grid


class EnvModelIntegrationTest(unittest.TestCase):
    def test_env_and_model_rollout(self):
        env = _build_small_env(max_steps=100)
        rng = jax.random.PRNGKey(0)
        state = env.env_reset(rng, train=True)

        model = PerceiverActorCritic()
        dummy_task, dummy_ids, dummy_canvas = _make_model_inputs(state)
        params = model.init(jax.random.PRNGKey(1), dummy_task, dummy_ids, dummy_canvas)

        @jax.jit
        def policy_fn(params, rng, task_grids, grid_ids, canvas):
            outputs = model.apply(params, task_grids, grid_ids, canvas, deterministic=True)
            pooled_logits = outputs["logits"].mean(axis=1)  # (B, policy_dim)
            actions = jax.random.categorical(rng, pooled_logits, axis=-1)
            return actions.astype(jnp.int32), outputs["value"]

        steps_taken = 0
        total_reward = 0.0
        for _ in range(env.max_steps):
            task_grids, grid_ids, canvas = _make_model_inputs(state)
            state_rng, action_rng = jax.random.split(state.rng)
            state = replace(state, rng=state_rng)
            actions, values = policy_fn(params, action_rng, task_grids, grid_ids, canvas)
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
