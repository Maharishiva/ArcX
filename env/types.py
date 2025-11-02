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
    (jit, vmap, pmap, grad, etc.). The environment keeps additional metadata
    that is required for reward shaping and for reconstructing the full set of
    demonstrations for a sampled ARC task.
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
    problem_idx: jnp.ndarray
    selected_color: jnp.ndarray
    last_action: jnp.ndarray
    baseline_score: jnp.ndarray
    prev_progress: jnp.ndarray

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
            self.problem_idx,
            self.selected_color,
            self.last_action,
            self.baseline_score,
            self.prev_progress,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple) -> "ARCEnvState":
        """Reconstruct state from flattened pytree."""
        return cls(*children)
