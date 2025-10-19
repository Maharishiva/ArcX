"""Wrappers and observation utilities for the ARC environment."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from .env import ARCEnv
from .types import ARCEnvState

__all__ = [
    "make_batched_reset",
    "make_batched_step",
    "observe",
    "observe_compact",
]


def observe(state: ARCEnvState, include_done_channel: bool = True) -> jnp.ndarray:
    """Return a multi-channel observation tensor for agents.

    Stacks channels: canvas, input, target, cursor_mask (1 at cursor),
    and optionally a done mask.
    
    Parameters
    ----------
    state : ARCEnvState
        Current environment state
    include_done_channel : bool
        If True, include a done mask channel (default: True)
    
    Returns
    -------
    jnp.ndarray
        Observation tensor with shape (5, 30, 30) if include_done_channel=True,
        otherwise (4, 30, 30)
    """
    cursor_mask = jnp.zeros_like(state.canvas)
    cursor_mask = cursor_mask.at[state.cursor[0], state.cursor[1]].set(1)
    
    channels = [
        state.canvas,
        state.inp,
        state.target,
        cursor_mask,
    ]
    
    if include_done_channel:
        done_mask = jnp.full_like(state.canvas, state.done.astype(jnp.int32))
        channels.append(done_mask)
    
    obs = jnp.stack(channels, axis=0)
    return obs


def observe_compact(state: ARCEnvState) -> dict:
    """Return a compact observation dictionary with scalar done flag.
    
    This is more aligned with typical RL interfaces where done is a scalar
    rather than a broadcasted channel.
    
    Parameters
    ----------
    state : ARCEnvState
        Current environment state
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'grid': (4, 30, 30) stacked [canvas, input, target, cursor_mask]
        - 'cursor': (2,) cursor position
        - 'done': scalar bool
        - 'steps': scalar int32
    """
    cursor_mask = jnp.zeros_like(state.canvas)
    cursor_mask = cursor_mask.at[state.cursor[0], state.cursor[1]].set(1)
    
    grid = jnp.stack([
        state.canvas,
        state.inp,
        state.target,
        cursor_mask,
    ], axis=0)
    
    return {
        'grid': grid,
        'cursor': state.cursor,
        'done': state.done,
        'steps': state.steps,
    }


def make_batched_reset(env: ARCEnv) -> Callable[[jax.Array, bool], ARCEnvState]:
    """Return a vmapped reset over batch of rng keys.

    Args:
        env: The environment instance.

    Returns:
        A function (rng_keys, train) -> batched ARCEnvState
    """

    def _reset(rng: jax.Array, train: bool = True) -> ARCEnvState:
        return env.env_reset(rng, train=train)

    return jax.vmap(_reset, in_axes=(0, None))


def make_batched_step(env: ARCEnv) -> Callable[[ARCEnvState, jnp.ndarray], Tuple[ARCEnvState, jnp.ndarray, jnp.ndarray]]:
    """Return a vmapped step over a batch of states and actions.

    Args:
        env: The environment instance.

    Returns:
        A function (states, actions) -> (next_states, rewards, dones)
    """

    def _step(state: ARCEnvState, action: jnp.ndarray):
        return env.env_step(state, action)

    return jax.vmap(_step, in_axes=(0, 0))

