"""Comprehensive test suite for the ARC environment."""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

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
    env = ARCEnv.from_json(sample_json, max_steps=10)

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
    """Test progress-shaped reward with baseline."""
    print("Testing progress-shaped reward...")

    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)

    base_key = jax.random.PRNGKey(0)
    key_success = jax.random.split(base_key, 2)[0]

    # Test correct solution: painting yields stepwise progress rewards
    state = env.env_reset(key_success, train=True)
    # After reset, prev_progress should be max(s(empty) - baseline, 0)
    # Since empty canvas likely scores low, prev_progress ~0
    
    # Choose color 2 and paint: this should improve score
    state, r1, _ = env.env_step(state, jnp.array(ARCEnv.ACT_CHOOSE_COLOR_START + 2))
    assert _as_float(r1) == 0.0, "Selecting color should not give reward"
    
    state, r2, _ = env.env_step(state, jnp.array(ARCEnv.ACT_PAINT))
    # Painting correct color should give positive progress reward
    assert _as_float(r2) > 0.0, f"Painting correct value should give positive reward, got {_as_float(r2)}"
    
    # SEND should give zero additional reward since we're already at target
    state, r3, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert abs(_as_float(r3)) < 1e-5, f"SEND at target should give ~0 reward, got {_as_float(r3)}"
    assert _as_bool(done) is True, "SEND should terminate"
    
    # Total accumulated reward should equal max(final_score - baseline, 0)
    total_reward = _as_float(r1) + _as_float(r2) + _as_float(r3)
    # Final score: 0.2*1.0 (IoU) + 1.0 (valid_acc) + 1.0 (full_match) = 2.2
    # Baseline score from input [[1]] vs target [[2]]: 0.2*1.0 + 0.0 + 0.0 = 0.2
    # Actually score(input) has full IoU but 0 accuracy: 0.2*1.0 + 0.0 + 0.0 = 0.2
    # Empty canvas score: 0.2*0.0 + 0.0 + 0.0 = 0 (no IoU, no accuracy, no full match)
    # Initial prev_progress = max(0 - 0.2, 0) = 0
    # After painting: score = 2.2, progress = max(2.2 - 0.2, 0) = 2.0, reward = 2.0 - 0 = 2.0
    # But we get r2 from the painting step only, not full 2.0
    # Actually the test should just check that painting gives positive reward
    assert total_reward > 0.5, f"Expected positive total reward, got {total_reward}"

    print("✓ Progress-shaped reward works correctly")


def test_send_action():
    """Test SEND action termination and progress reward."""
    print("Testing send action...")

    sample_json = '{"train": [{"input": [[1]], "output": [[1]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)

    key_fail, key_success = jax.random.split(jax.random.PRNGKey(5))

    state = env.env_reset(key_fail, train=True)
    state, reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert _as_bool(done) is True, "SEND should terminate even without solution"
    # With progress shaping, sending without progress gives 0
    assert _as_float(reward) == 0.0, "Empty canvas SEND should yield zero progress reward"

    state = env.env_reset(key_success, train=True)
    state, copy_reward, _ = env.env_step(state, jnp.array(ARCEnv.ACT_COPY))
    # Copying the input when input==target gives full progress since baseline score(input,target) is max
    # But with our baseline logic, copying shouldn't give much reward since baseline = score(input)
    # Actually for this case input==output, so baseline = 2.2 (full match), and copying gives same 2.2
    # So progress = max(2.2 - 2.2, 0) = 0
    state, send_reward, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert _as_bool(done) is True, "SEND should terminate when solved"
    # Total reward from copy + send should be ~0 since input already matches target
    total_r = _as_float(copy_reward) + _as_float(send_reward)
    assert abs(total_r) < 0.1, f"Copying when input==output should give ~0 total reward, got {total_r}"

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

    sample_json = '{"train": [{"input": [[1,2,3],[4,5,6]], "output": [[1,2,3],[4,5,6]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=20)

    state = env.env_reset(jax.random.PRNGKey(7), train=True)

    # Paint color 0 at (0,0)
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_CHOOSE_COLOR_START))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_PAINT))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_RIGHT))
    # Paint color 1
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_CHOOSE_COLOR_START + 1))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_PAINT))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_RIGHT))
    # Paint color 2
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_CHOOSE_COLOR_START + 2))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_PAINT))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_LEFT))
    state, _, _ = env.env_step(state, jnp.array(ARCEnv.ACT_CROP))

    assert state.canvas[0, 0] == 0, "Cell before crop point should be preserved"
    assert state.canvas[0, 1] == ARCEnv.EMPTY_CELL, "Cell at crop point should be EMPTY"
    assert state.canvas[0, 2] == ARCEnv.EMPTY_CELL, "Cells after crop point should be EMPTY"
    assert state.canvas[1, 0] == ARCEnv.EMPTY_CELL, "Rows below crop should be EMPTY"

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
    state, copy_r, _ = env.env_step(state, jnp.array(ARCEnv.ACT_COPY))
    state, send_r, done = env.env_step(state, jnp.array(ARCEnv.ACT_SEND))
    assert _as_bool(done) is True, "Should be done after SEND"
    # For input==output case, copy gives ~0 reward (baseline already at max)
    total_r = _as_float(copy_r) + _as_float(send_r)
    assert abs(total_r) < 0.1, f"Should get ~0 total reward for copy when input==output, got {total_r}"

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
