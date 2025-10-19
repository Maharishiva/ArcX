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
    background_color: int = 0
) -> jnp.ndarray:
    """Shift grid so that (row_offset, col_offset) becomes (0, 0).
    
    Cells that were -1 (padding) will be replaced with background_color.
    New padding areas created by the shift are filled with background_color.
    
    Parameters
    ----------
    grid : jnp.ndarray
        The grid to shift (grid_size × grid_size)
    row_offset : int
        Row position that should move to row 0
    col_offset : int
        Col position that should move to col 0
    grid_size : int
        Size of the grid (assumes square)
    empty_value : int
        Sentinel value for empty cells (default: -1)
    background_color : int
        Color to use for padding after shift (default: 0)
    
    Returns
    -------
    jnp.ndarray
        Shifted grid with new padding
    """
    # Replace -1 with background color in the original grid
    grid_with_bg = jnp.where(grid == empty_value, background_color, grid)
    
    # Create coordinate arrays for source positions
    rows = jnp.arange(grid_size)
    cols = jnp.arange(grid_size)
    
    # Calculate source positions (where to read from)
    src_rows = rows + row_offset
    src_cols = cols + col_offset
    
    # Create meshgrid for indexing
    src_row_grid, src_col_grid = jnp.meshgrid(src_rows, src_cols, indexing='ij')
    
    # Check which positions are valid (within bounds of original grid)
    valid = (src_row_grid >= 0) & (src_row_grid < grid_size) & \
            (src_col_grid >= 0) & (src_col_grid < grid_size)
    
    # Clip indices to valid range for safe indexing
    src_row_grid_safe = jnp.clip(src_row_grid, 0, grid_size - 1)
    src_col_grid_safe = jnp.clip(src_col_grid, 0, grid_size - 1)
    
    # Gather values from source positions
    shifted = grid_with_bg[src_row_grid_safe, src_col_grid_safe]
    
    # Fill invalid positions (out of bounds) with background color
    shifted = jnp.where(valid, shifted, background_color)
    
    return shifted 