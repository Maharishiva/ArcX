"""Comprehensive test suite for the ARC environment."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import ARCEnv
from env.wrappers import make_batched_reset, make_batched_step, observe, observe_compact


def _as_float(x):
    return float(jnp.asarray(x))


def _as_bool(x):
    return bool(jnp.asarray(x))


def test_basic_reset_and_step():
    """Test basic reset and step functionality."""
    print("Testing basic reset and step...")

    sample_json = (
        '{"train": [{"input": [[1,2],[3,4]], "output": [[5,6],[7,8]]}], "test": []}'
    )
    env = ARCEnv.from_json(sample_json, max_steps=10, reward_mode="sparse")

    rng = jax.random.PRNGKey(42)
    state = env.env_reset(rng, train=True)

    assert state.canvas.shape == (30, 30), "Canvas should be 30x30"
    assert state.cursor.shape == (2,), "Cursor should be 2D"
    assert jnp.all(state.cursor == jnp.array([0, 0])), "Cursor should start at (0,0)"
    assert state.steps == 0, "Steps should start at 0"
    assert not _as_bool(state.done), "Should not be done initially"

    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_DOWN))
    assert jnp.all(state.cursor == jnp.array([1, 0])), "Cursor should move down"
    assert _as_bool(done) is False, "Should not be done after movement"
    assert _as_float(reward) == 0.0, "Movement should not yield reward"

    # Choose color 1, then paint it
    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_CHOOSE_COLOR_START + 1))
    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_PAINT))
    assert state.canvas[1, 0] == 1, "Should paint colour 1"
    assert _as_float(reward) == 0.0, "Painting should not yield immediate reward in sparse mode"
    assert _as_bool(done) is False, "Painting should not terminate"

    print("✓ Basic reset and step work correctly")


def test_masked_reward():
    """Test that rewards are only granted on SEND and only for valid cells."""
    print("Testing masked reward...")

    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10, reward_mode="dense")

    base_key = jax.random.PRNGKey(0)
    key_send_only, key_success, key_padded = jax.random.split(base_key, 3)

    # Sending without matching target yields zero reward but terminates.
    state = env.env_reset(key_send_only, train=True)
    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert _as_float(reward) == 0.0, "SEND without solution should have zero reward"
    assert _as_bool(done) is True, "SEND should terminate the episode"

    # Painting the correct value in the valid region and then sending yields reward.
    state = env.env_reset(key_success, train=True)
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_CHOOSE_COLOR_START + 2))
    state, paint_reward, _ = env.env_step(state, jnp.array(ARCEnv.ACT_PAINT))
    assert abs(_as_float(paint_reward) - 2.2) < 1e-5, "Painting correct cell should yield shaped reward"
    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert _as_float(reward) == 0.0, "SEND after solving should not add extra reward"
    assert _as_bool(done) is True, "SEND should terminate"

    # Painting in a padded region should not affect reward when sending.
    state = env.env_reset(key_padded, train=True)
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_DOWN))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_RIGHT))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_CHOOSE_COLOR_START))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_PAINT))
    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert _as_float(reward) == 0.0, "Padded-region changes should not count toward reward"
    assert _as_bool(done) is True, "SEND should terminate"

    print("✓ Masked reward works correctly")


def test_send_action():
    """Test SEND action termination and reward semantics."""
    print("Testing send action...")

    sample_json = '{"train": [{"input": [[1]], "output": [[1]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10, reward_mode="dense")

    key_fail, key_success = jax.random.split(jax.random.PRNGKey(5))

    state = env.env_reset(key_fail, train=True)
    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert _as_bool(done) is True, "SEND should terminate even without solution"
    assert _as_float(reward) == 0.0, "Incorrect canvas should yield zero reward"

    state = env.env_reset(key_success, train=True)
    state, reward_copy, _ = env.env_step(state, jnp.array(ARCEnv.ACT_COPY))
    assert abs(_as_float(reward_copy) - 2.2) < 1e-5, "Copying baseline should yield the full score jump"
    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert _as_bool(done) is True, "SEND should terminate when solved"
    assert _as_float(reward) == 0.0, "Final reward is emitted via progress steps"

    print("✓ Send action works correctly")


def test_copy_full_canvas():
    """Test that COPY duplicates the entire input grid."""
    print("Testing copy action...")

    sample_json = '{"train": [{"input": [[1,2],[3,4]], "output": [[9,9],[9,9]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)

    state = env.env_reset(jax.random.PRNGKey(123), train=True)
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_COPY))
    assert jnp.array_equal(state.canvas, state.inp), "COPY should clone the whole input grid"

    print("✓ Copy action works correctly")


def test_crop_action():
    """Test crop action (action 16)."""
    print("Testing crop action...")

    env = ARCEnv.from_json('{"train": [], "test": [{"input": [[0]], "output": [[0]]}]}', max_steps=20)

    state = env.env_reset(jax.random.PRNGKey(7), train=False)
    base_canvas = jnp.full((env.GRID_SIZE, env.GRID_SIZE), ARCEnv.EMPTY_CELL, dtype=jnp.int32)
    pattern = jnp.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
        ],
        dtype=jnp.int32,
    )
    base_canvas = base_canvas.at[:3, :3].set(pattern)
    cursor = jnp.array([1, 1], dtype=jnp.int32)

    state = replace(state, canvas=base_canvas, cursor=cursor)
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_CROP))

    expected = jnp.array(
        [
            [0, 1, ARCEnv.EMPTY_CELL],
            [1, 2, ARCEnv.EMPTY_CELL],
            [ARCEnv.EMPTY_CELL, ARCEnv.EMPTY_CELL, ARCEnv.EMPTY_CELL],
        ],
        dtype=jnp.int32,
    )

    assert jnp.array_equal(state.canvas[:3, :3], expected), "Crop should preserve quadrant up-left of cursor"
    assert jnp.array_equal(state.cursor, cursor), "Cursor position should remain unchanged"

    print("✓ Crop action works correctly")


def test_shift_to_origin():
    """Test move-to-origin action (action 17)."""
    print("Testing shift-to-origin action...")

    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=30)

    state = env.env_reset(jax.random.PRNGKey(11), train=True)

    for _ in range(3):
        state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_DOWN))
    for _ in range(5):
        state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_RIGHT))

    assert jnp.all(state.cursor == jnp.array([3, 5])), "Cursor should be at (3,5)"

    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_CHOOSE_COLOR_START + 3))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_PAINT))
    assert state.canvas[3, 5] == 3, "Should have colour 3 at (3,5)"

    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_MOVE_TO_ORIGIN))

    assert jnp.all(state.cursor == jnp.array([0, 0])), "Cursor should move to origin"
    assert state.canvas[0, 0] == 3, "Colour should shift to origin"
    assert state.canvas[3, 5] == ARCEnv.BACKGROUND_COLOR, "Old position should be background colour"

    print("✓ Shift-to-origin action works correctly")


def test_observation_functions():
    """Test observation wrapper functions."""
    print("Testing observation functions...")

    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)

    state = env.env_reset(jax.random.PRNGKey(13), train=True)

    obs = observe(state, include_done_channel=True)
    assert obs.shape == (5, 30, 30), "Full obs should be (5, 30, 30)"

    obs_no_done = observe(state, include_done_channel=False)
    assert obs_no_done.shape == (4, 30, 30), "Obs without done should be (4, 30, 30)"

    obs_compact = observe_compact(state)
    assert "grid" in obs_compact and "cursor" in obs_compact and "done" in obs_compact, "Missing keys in compact obs"
    assert obs_compact["grid"].shape == (4, 30, 30), "Grid should be (4, 30, 30)"

    print("✓ Observation functions work correctly")


def test_batched_operations():
    """Test vmapped batched reset and step."""
    print("Testing batched operations...")

    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)

    batch_size = 4
    rngs = jax.random.split(jax.random.PRNGKey(0), batch_size)
    batched_reset = make_batched_reset(env)
    states = batched_reset(rngs, True)

    assert states.canvas.shape == (batch_size, 30, 30), "Batched canvas should be (B, 30, 30)"
    assert states.cursor.shape == (batch_size, 2), "Batched cursor should be (B, 2)"

    batched_step = make_batched_step(env)
    actions = jnp.array([ARCEnv.ACT_DOWN, ARCEnv.ACT_LEFT, ARCEnv.ACT_RIGHT, ARCEnv.ACT_CHOOSE_COLOR_START])
    states, rewards, dones = batched_step(states, actions)

    assert rewards.shape == (batch_size,), "Batched rewards should be (B,)"
    assert dones.shape == (batch_size,), "Batched dones should be (B,)"

    print("✓ Batched operations work correctly")


def test_empty_dataset_error():
    """Test that empty datasets raise appropriate errors."""
    print("Testing empty dataset handling...")

    sample_json = '{"train": [], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)

    rng = jax.random.PRNGKey(42)

    try:
        env.env_reset(rng, train=True)
        assert False, "Should have raised ValueError for empty train set"
    except ValueError as e:
        assert "empty" in str(e).lower(), "Error should mention empty dataset"

    try:
        env.env_reset(rng, train=False)
        assert False, "Should have raised ValueError for empty test set"
    except ValueError as e:
        assert "empty" in str(e).lower(), "Error should mention empty dataset"

    print("✓ Empty dataset errors work correctly")


def test_action_clamping():
    """Test that invalid actions are clamped."""
    print("Testing action clamping...")

    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)

    state = env.env_reset(jax.random.PRNGKey(21), train=True)

    try:
        env.env_step(state, jnp.array(100))
        print("✓ Action clamping works correctly")
    except Exception as e:
        print(f"✗ Action clamping failed: {e}")
        raise


def test_termination_conditions():
    """Test episode termination conditions."""
    print("Testing termination conditions...")

    sample_json = '{"train": [{"input": [[1]], "output": [[1]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)

    key_solve, key_budget = jax.random.split(jax.random.PRNGKey(99))

    state = env.env_reset(key_solve, train=True)
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_COPY))
    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert _as_bool(done) is True, "Should be done after SEND"
    assert abs(_as_float(reward) - 2.2) < 1e-5, "Should get final score as reward for solved SEND"

    env2 = ARCEnv.from_json(sample_json, max_steps=2)
    state2 = env2.env_reset(key_budget, train=True)
    state2, _, done = env2.env_step(state2, jnp.array(ARCEnv.ACT_UP))
    assert _as_bool(done) is False, "Should not be done after 1 step"
    state2, _, done = env2.env_step(state2, jnp.array(ARCEnv.ACT_UP))
    assert _as_bool(done) is True, "Should be done after hitting max_steps"

    print("✓ Termination conditions work correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running ARC Environment Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_basic_reset_and_step,
        test_masked_reward,
        test_send_action,
        test_copy_full_canvas,
        test_crop_action,
        test_shift_to_origin,
        test_observation_functions,
        test_batched_operations,
        test_empty_dataset_error,
        test_action_clamping,
        test_termination_conditions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    raise SystemExit(0 if success else 1)
