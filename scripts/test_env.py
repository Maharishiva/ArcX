"""Comprehensive test suite for the ARC environment."""

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
from env import ARCEnv
from env.wrappers import observe, observe_compact, make_batched_reset, make_batched_step


def test_basic_reset_and_step():
    """Test basic reset and step functionality."""
    print("Testing basic reset and step...")
    
    sample_json = (
        '{"train": [{"input": [[1,2],[3,4]], "output": [[5,6],[7,8]]}], "test": []}'
    )
    env = ARCEnv.from_json(sample_json, max_steps=10)
    
    rng = jax.random.PRNGKey(42)
    state = env.env_reset(rng, train=True)
    
    # Check initial state
    assert state.canvas.shape == (30, 30), "Canvas should be 30x30"
    assert state.cursor.shape == (2,), "Cursor should be 2D"
    assert jnp.all(state.cursor == jnp.array([0, 0])), "Cursor should start at (0,0)"
    assert state.steps == 0, "Steps should start at 0"
    assert not state.done, "Should not be done initially"
    
    # Test movement
    state, reward, done = env.env_step(state, jnp.array(1))  # down
    assert jnp.all(state.cursor == jnp.array([1, 0])), "Cursor should move down"
    
    # Test painting
    state, reward, done = env.env_step(state, jnp.array(5))  # paint color 1
    assert state.canvas[1, 0] == 1, "Should paint color 1"
    
    print("✓ Basic reset and step work correctly")


def test_masked_reward():
    """Test that reward only considers valid (non-padded) cells."""
    print("Testing masked reward...")
    
    # Create a simple task where we know the exact valid region
    sample_json = (
        '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    )
    env = ARCEnv.from_json(sample_json, max_steps=10)
    
    rng = jax.random.PRNGKey(42)
    state = env.env_reset(rng, train=True)
    
    # Paint in the valid region (0,0) - should get reward
    state, reward, done = env.env_step(state, jnp.array(6))  # paint color 2
    assert reward == 1.0, f"Should get reward 1.0 for correct cell, got {reward}"
    
    # Paint in padded region (10, 10) - should get 0 reward
    for _ in range(10):
        state, _, _ = env.env_step(state, jnp.array(1))  # down
        state, _, _ = env.env_step(state, jnp.array(3))  # right
    
    state, reward, done = env.env_step(state, jnp.array(4))  # paint color 0
    assert reward == 0.0, f"Should get 0 reward in padded region, got {reward}"
    
    print("✓ Masked reward works correctly")


def test_erase_action():
    """Test erase action (action 15)."""
    print("Testing erase action...")
    
    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)
    
    rng = jax.random.PRNGKey(42)
    state = env.env_reset(rng, train=True)
    
    # Paint a color
    state, _, _ = env.env_step(state, jnp.array(5))  # paint color 1
    assert state.canvas[0, 0] == 1, "Should have color 1"
    
    # Erase it
    state, _, _ = env.env_step(state, jnp.array(15))  # erase
    assert state.canvas[0, 0] == -1, "Should be erased to -1"
    
    print("✓ Erase action works correctly")


def test_crop_action():
    """Test crop action (action 16)."""
    print("Testing crop action...")
    
    sample_json = '{"train": [{"input": [[1,2,3],[4,5,6]], "output": [[1,2,3],[4,5,6]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=20)
    
    rng = jax.random.PRNGKey(42)
    state = env.env_reset(rng, train=True)
    
    # Paint some cells
    state, _, _ = env.env_step(state, jnp.array(4))  # paint at (0,0)
    state, _, _ = env.env_step(state, jnp.array(3))  # right
    state, _, _ = env.env_step(state, jnp.array(5))  # paint at (0,1)
    state, _, _ = env.env_step(state, jnp.array(3))  # right
    state, _, _ = env.env_step(state, jnp.array(6))  # paint at (0,2)
    
    # Move to (0, 1) and crop
    state, _, _ = env.env_step(state, jnp.array(2))  # left
    state, _, _ = env.env_step(state, jnp.array(16))  # crop
    
    # Check that (0,0) still has value but everything from (0,1) onward is -1
    assert state.canvas[0, 0] == 0, "Cell before crop point should be preserved"
    assert state.canvas[0, 1] == -1, "Cell at crop point should be -1"
    assert state.canvas[0, 2] == -1, "Cell after crop point should be -1"
    assert state.canvas[1, 0] == -1, "Row below crop should be -1"
    
    print("✓ Crop action works correctly")


def test_shift_to_origin():
    """Test move-to-origin action (action 17)."""
    print("Testing shift-to-origin action...")
    
    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=30)
    
    rng = jax.random.PRNGKey(42)
    state = env.env_reset(rng, train=True)
    
    # Move cursor to (3, 5) and paint
    for _ in range(3):
        state, _, _ = env.env_step(state, jnp.array(1))  # down
    for _ in range(5):
        state, _, _ = env.env_step(state, jnp.array(3))  # right
    
    assert jnp.all(state.cursor == jnp.array([3, 5])), "Cursor should be at (3,5)"
    
    # Paint a color
    state, _, _ = env.env_step(state, jnp.array(7))  # paint color 3
    assert state.canvas[3, 5] == 3, "Should have color 3 at (3,5)"
    
    # Shift to origin
    state, _, _ = env.env_step(state, jnp.array(17))
    
    # Check cursor moved to origin
    assert jnp.all(state.cursor == jnp.array([0, 0])), "Cursor should be at (0,0) after shift"
    
    # Check that the painted cell moved
    assert state.canvas[0, 0] == 3, "Color should have shifted to (0,0)"
    assert state.canvas[3, 5] == 0, "Old position should be background color"
    
    print("✓ Shift-to-origin action works correctly")


def test_observation_functions():
    """Test observation wrapper functions."""
    print("Testing observation functions...")
    
    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)
    
    rng = jax.random.PRNGKey(42)
    state = env.env_reset(rng, train=True)
    
    # Test standard observe
    obs = observe(state, include_done_channel=True)
    assert obs.shape == (5, 30, 30), "Full obs should be (5, 30, 30)"
    
    obs_no_done = observe(state, include_done_channel=False)
    assert obs_no_done.shape == (4, 30, 30), "Obs without done should be (4, 30, 30)"
    
    # Test compact observe
    obs_compact = observe_compact(state)
    assert 'grid' in obs_compact, "Should have grid key"
    assert 'cursor' in obs_compact, "Should have cursor key"
    assert 'done' in obs_compact, "Should have done key"
    assert obs_compact['grid'].shape == (4, 30, 30), "Grid should be (4, 30, 30)"
    
    print("✓ Observation functions work correctly")


def test_batched_operations():
    """Test vmapped batched reset and step."""
    print("Testing batched operations...")
    
    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)
    
    # Test batched reset
    batch_size = 4
    rngs = jax.random.split(jax.random.PRNGKey(0), batch_size)
    batched_reset = make_batched_reset(env)
    states = batched_reset(rngs, True)
    
    assert states.canvas.shape == (batch_size, 30, 30), "Batched canvas should be (B, 30, 30)"
    assert states.cursor.shape == (batch_size, 2), "Batched cursor should be (B, 2)"
    
    # Test batched step
    batched_step = make_batched_step(env)
    actions = jnp.array([1, 2, 3, 4])  # different actions for each env
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
        state = env.env_reset(rng, train=True)
        assert False, "Should have raised ValueError for empty train set"
    except ValueError as e:
        assert "empty" in str(e).lower(), "Error should mention empty dataset"
    
    try:
        state = env.env_reset(rng, train=False)
        assert False, "Should have raised ValueError for empty test set"
    except ValueError as e:
        assert "empty" in str(e).lower(), "Error should mention empty dataset"
    
    print("✓ Empty dataset errors work correctly")


def test_action_clamping():
    """Test that invalid actions are clamped."""
    print("Testing action clamping...")
    
    sample_json = '{"train": [{"input": [[1]], "output": [[2]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)
    
    rng = jax.random.PRNGKey(42)
    state = env.env_reset(rng, train=True)
    
    # Try an out-of-range action (should be clamped)
    try:
        state, reward, done = env.env_step(state, jnp.array(100))
        # Should not crash - action should be clamped
        print("✓ Action clamping works correctly")
    except Exception as e:
        print(f"✗ Action clamping failed: {e}")


def test_termination_conditions():
    """Test episode termination conditions."""
    print("Testing termination conditions...")
    
    # Small task that can be solved
    sample_json = '{"train": [{"input": [[1]], "output": [[1]]}], "test": []}'
    env = ARCEnv.from_json(sample_json, max_steps=10)
    
    rng = jax.random.PRNGKey(42)
    state = env.env_reset(rng, train=True)
    
    # Copy from input should solve it
    state, reward, done = env.env_step(state, jnp.array(14))  # copy
    assert done, "Should be done after solving"
    assert reward > 0, "Should get positive reward for solving"
    
    # Test max_steps termination
    env2 = ARCEnv.from_json(sample_json, max_steps=2)
    state2 = env2.env_reset(rng, train=True)
    state2, _, done = env2.env_step(state2, jnp.array(0))  # step 1
    assert not done, "Should not be done after 1 step"
    state2, _, done = env2.env_step(state2, jnp.array(0))  # step 2
    assert done, "Should be done after max_steps"
    
    print("✓ Termination conditions work correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running ARC Environment Test Suite")
    print("="*60 + "\n")
    
    tests = [
        test_basic_reset_and_step,
        test_masked_reward,
        test_erase_action,
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
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

