import jax
import jax.numpy as jnp

__all__ = ["pad_to_30", "compute_valid_mask", "shift_grid_to_origin"]


def pad_to_30(arr):
    """Pad a 2-D Python list up to 30×30 using the sentinel value -1.

    Parameters
    ----------
    arr : list[list[int]]
        A nested list representing the ARC grid.

    Returns
    -------
    jnp.ndarray
        A `(30, 30)` int32 array with the original values in the top-left
        corner and `-1` elsewhere.
    """
    h, w = len(arr), len(arr[0])
    padded = jnp.full((30, 30), -1, dtype=jnp.int32)
    arr_jnp = jnp.array(arr, dtype=jnp.int32)
    return padded.at[:h, :w].set(arr_jnp)


def compute_valid_mask(grid: jnp.ndarray, empty_value: int = -1) -> jnp.ndarray:
    """Compute a boolean mask of valid (non-padded) cells.
    
    Parameters
    ----------
    grid : jnp.ndarray
        Grid array (typically 30×30)
    empty_value : int
        Sentinel value used for padding (default: -1)
    
    Returns
    -------
    jnp.ndarray
        Boolean array of same shape as grid, True where cell is valid
    """
    return grid != empty_value


def shift_grid_to_origin(
    grid: jnp.ndarray,
    row_offset: int,
    col_offset: int,
    grid_size: int,
    empty_value: int = -1,
    background_color: int = 0,
) -> jnp.ndarray:
    """Shift grid so that (row_offset, col_offset) becomes (0, 0).

    This version is fully JIT-friendly:
    - Replaces EMPTY cells with background in the source
    - Writes the source into a larger padded canvas at a *static* position
    - Takes a `dynamic_slice` window of static size (grid_size, grid_size)
    """
    grid_bg = jnp.where(grid == empty_value, background_color, grid)

    big = jnp.full((grid_size * 2, grid_size * 2), background_color, dtype=grid.dtype)

    insert_r = grid_size - row_offset
    insert_c = grid_size - col_offset
    big = jax.lax.dynamic_update_slice(big, grid_bg, (insert_r, insert_c))

    start_r = grid_size
    start_c = grid_size

    return jax.lax.dynamic_slice(big, (start_r, start_c), (grid_size, grid_size))
