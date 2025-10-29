"""Type definitions for the ARC environment."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

__all__ = ["ARCEnvState"]


@jax.tree_util.register_pytree_node_class
@dataclass
class ARCEnvState:
    """Immutable state container for the ARC environment.
    
    All fields are JAX arrays for full compatibility with JAX transformations
    (jit, vmap, pmap, grad, etc.).
    
    Attributes
    ----------
    rng : jnp.ndarray
        JAX PRNG key for random operations
    canvas : jnp.ndarray
        Agent's working canvas (30, 30) int32 array
    cursor : jnp.ndarray
        Cursor position [row, col] as (2,) int32 array
    inp : jnp.ndarray
        Input grid for current task (30, 30) int32 array
    target : jnp.ndarray
        Target output grid for current task (30, 30) int32 array
    valid_mask : jnp.ndarray
        Boolean mask where target != EMPTY_CELL; cached for reward/done checks
    done : jnp.ndarray
        Scalar bool indicating episode termination
    steps : jnp.ndarray
        Scalar int32 counting steps taken in episode
    episode_idx : jnp.ndarray
        Scalar int32 index of current task in dataset
    Additional fields for control and observation:
    selected_color : jnp.ndarray
        Scalar int32, currently selected paint color in [0, 9]
    last_action : jnp.ndarray
        Scalar int32, last action taken (or -1 before any action)
    """

    rng: jnp.ndarray
    canvas: jnp.ndarray
    cursor: jnp.ndarray
    inp: jnp.ndarray
    target: jnp.ndarray
    valid_mask: jnp.ndarray
    done: jnp.ndarray
    steps: jnp.ndarray
    episode_idx: jnp.ndarray
    selected_color: jnp.ndarray
    last_action: jnp.ndarray

    def tree_flatten(self):
        """Flatten state for JAX pytree operations."""
        children = (
            self.rng,
            self.canvas,
            self.cursor,
            self.inp,
            self.target,
            self.valid_mask,
            self.done,
            self.steps,
            self.episode_idx,
            self.selected_color,
            self.last_action,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple) -> "ARCEnvState":
        """Reconstruct state from flattened pytree."""
        return cls(*children)
