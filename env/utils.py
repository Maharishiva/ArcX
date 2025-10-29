import jax
import jax.numpy as jnp

__all__ = ["pad_to_30", "compute_valid_mask", "shift_grid_to_origin", "flood_fill_n4"]


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


def _n4(mask: jnp.ndarray) -> jnp.ndarray:
    """4-neighborhood dilation of a boolean mask.

    Returns a mask that includes up/down/left/right neighbors of True cells.
    """
    up = jnp.pad(mask[1:, :], ((0, 1), (0, 0)))
    down = jnp.pad(mask[:-1, :], ((1, 0), (0, 0)))
    left = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)))
    right = jnp.pad(mask[:, :-1], ((0, 0), (1, 0)))
    return up | down | left | right


@jax.jit
def flood_fill_n4(grid: jnp.ndarray, cursor_rc: jnp.ndarray, new_color: jnp.ndarray) -> jnp.ndarray:
    """Flood-fill from cursor using 4-neighborhood and replace with new_color.

    Parameters
    ----------
    grid : jnp.ndarray
        (H, W) int32 grid to be filled
    cursor_rc : jnp.ndarray
        (2,) int32 array [row, col] indicating seed position
    new_color : jnp.ndarray
        Scalar int32 color to write into the filled region
    """
    g = grid.astype(jnp.int32)
    h, w = g.shape
    r = jnp.clip(cursor_rc[0], 0, h - 1)
    c = jnp.clip(cursor_rc[1], 0, w - 1)

    target = g[r, c]
    same = g == target
    seed = (jnp.arange(h)[:, None] == r) & (jnp.arange(w)[None, :] == c)

    def cond_fun(state):
        visited, frontier = state
        del visited
        return jnp.any(frontier)

    def body_fun(state):
        visited, frontier = state
        new_frontier = _n4(frontier) & same & ~visited
        return visited | new_frontier, new_frontier

    visited, _ = jax.lax.while_loop(cond_fun, body_fun, (seed, seed))
    return jnp.where(visited, jnp.asarray(new_color, g.dtype), g)
