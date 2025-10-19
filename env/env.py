import json
import functools

import jax
import jax.numpy as jnp

from .utils import pad_to_30, compute_valid_mask, shift_grid_to_origin
from .types import ARCEnvState

__all__ = ["ARCEnv"]


class ARCEnv:
    """A JAX-compatible environment tailored for ARC grid tasks.

    Actions:
        0: move cursor up
        1: move cursor down
        2: move cursor left
        3: move cursor right
        4-13: paint the cell with colour (action-4) ∈ [0-9]
        14: copy value from *input* grid at cursor to canvas
        15: erase (paint -1 at cursor)
        16: crop - fill everything right and down from cursor with -1
        17: move-to-origin - shift grid so cursor moves to (0,0), pad with background color 0
    """

    # Hard-coded grid size used throughout the original ARC dataset
    GRID_SIZE: int = 30

    # Default episode length (can be overridden in constructor)
    DEFAULT_MAX_STEPS: int = 100

    NUM_ACTIONS: int = 18  # 0-17 inclusive
    
    # Sentinel value for padding/empty cells
    EMPTY_CELL: int = -1
    
    # Default background color for shift operation
    BACKGROUND_COLOR: int = 0

    # Action constants (public)
    ACT_UP: int = 0
    ACT_DOWN: int = 1
    ACT_LEFT: int = 2
    ACT_RIGHT: int = 3
    ACT_PAINT_START: int = 4   # inclusive
    ACT_PAINT_END: int = 13    # inclusive
    ACT_COPY: int = 14
    ACT_ERASE: int = 15
    ACT_CROP: int = 16
    ACT_MOVE_TO_ORIGIN: int = 17

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self,
                 train_input: jnp.ndarray,
                 train_output: jnp.ndarray,
                 test_input: jnp.ndarray,
                 test_output: jnp.ndarray,
                 max_steps: int = DEFAULT_MAX_STEPS):
        self.train_input = train_input  # shape [N_train, 30, 30]
        self.train_output = train_output
        self.test_input = test_input    # shape [N_test, 30, 30]
        self.test_output = test_output
        self.max_steps = max_steps

    # Convenient factory to load the canonical JSON representation
    @staticmethod
    def from_json(json_str: str, max_steps: int = DEFAULT_MAX_STEPS) -> "ARCEnv":
        """Create an environment from the raw ARC JSON string."""

        def _load_pairs(examples):
            ins, outs = [], []
            for e in examples:
                ins.append(pad_to_30(e["input"]))
                outs.append(pad_to_30(e["output"]))
            if len(ins) == 0:
                return (
                    jnp.zeros((0, ARCEnv.GRID_SIZE, ARCEnv.GRID_SIZE), dtype=jnp.int32),
                    jnp.zeros((0, ARCEnv.GRID_SIZE, ARCEnv.GRID_SIZE), dtype=jnp.int32),
                )
            return jnp.stack(ins), jnp.stack(outs)

        data = json.loads(json_str)
        tr_in, tr_out = _load_pairs(data.get("train", []))
        te_in, te_out = _load_pairs(data.get("test", []))
        return ARCEnv(tr_in, tr_out, te_in, te_out, max_steps)

    # (Note: ARCEnv itself is treated as static in jitted functions; no pytree needed.)

    # ------------------------------------------------------------------
    # Environment API (reset / step / render)
    # ------------------------------------------------------------------
    def env_reset(self, rng: jax.Array, train: bool = True) -> ARCEnvState:
        """Sample a random task and return an initial State object.
        
        Args:
            rng: JAX PRNG key
            train: If True, sample from train set; otherwise from test set
            
        Returns:
            Initial ARCEnvState
            
        Raises:
            ValueError: If the selected dataset is empty
        """
        rng, idx_rng = jax.random.split(rng)
        if train:
            n = self.train_input.shape[0]
            if n == 0:
                raise ValueError("Train dataset is empty. Cannot reset environment.")
            idx = jax.random.randint(idx_rng, (), 0, n)
            inp, target = self.train_input[idx], self.train_output[idx]
        else:
            n = self.test_input.shape[0]
            if n == 0:
                raise ValueError("Test dataset is empty. Cannot reset environment.")
            idx = jax.random.randint(idx_rng, (), 0, n)
            inp, target = self.test_input[idx], self.test_output[idx]

        canvas = jnp.full((self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32)

        return ARCEnvState(
            rng=rng,
            canvas=canvas,
            cursor=jnp.array([0, 0], dtype=jnp.int32),
            inp=inp,
            target=target,
            done=jnp.array(False),
            steps=jnp.array(0, dtype=jnp.int32),
            episode_idx=idx,
        )

    # JIT-compiled *functional* step method
    @functools.partial(jax.jit, static_argnums=(0,))
    def env_step(self, state: ARCEnvState, action: jnp.ndarray):
        """One interaction step – returns (next_state, reward, done).
        
        Args:
            state: Current environment state
            action: Action index (0-17)
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Clamp action to valid range to prevent jax.lax.switch errors
        action = jnp.clip(action, 0, self.NUM_ACTIONS - 1)

        # ──────────────────────────────────────────────────────────────────
        # Helper functions for cursor movement
        # ──────────────────────────────────────────────────────────────────
        def _up(c):
            return c.at[0].set(jnp.maximum(0, c[0] - 1))

        def _down(c):
            return c.at[0].set(jnp.minimum(self.GRID_SIZE - 1, c[0] + 1))

        def _left(c):
            return c.at[1].set(jnp.maximum(0, c[1] - 1))

        def _right(c):
            return c.at[1].set(jnp.minimum(self.GRID_SIZE - 1, c[1] + 1))

        # Consolidated action handler (lax.switch for branch-free JIT)
        def _apply(a, canvas, cursor):
            # Movement actions
            def act_up():
                return canvas, _up(cursor)

            def act_down():
                return canvas, _down(cursor)

            def act_left():
                return canvas, _left(cursor)

            def act_right():
                return canvas, _right(cursor)

            # Painting (actions 4-13)
            def paint():
                colour = a - 4
                new_canvas = canvas.at[cursor[0], cursor[1]].set(colour)
                return new_canvas, cursor

            # Copy from *input*
            def copy():
                val = state.inp[cursor[0], cursor[1]]
                new_canvas = canvas.at[cursor[0], cursor[1]].set(val)
                return new_canvas, cursor

            # Erase (paint -1)
            def erase():
                new_canvas = canvas.at[cursor[0], cursor[1]].set(self.EMPTY_CELL)
                return new_canvas, cursor

            # Crop: fill everything right and down from cursor with -1
            def crop():
                row, col = cursor[0], cursor[1]
                # Create mask for cells to crop (right and down from cursor)
                rows = jnp.arange(self.GRID_SIZE)
                cols = jnp.arange(self.GRID_SIZE)
                row_mask = rows >= row  # All rows from cursor down
                col_mask = cols >= col  # All cols from cursor right
                # Create 2D mask: true where we want to crop
                crop_mask = row_mask[:, None] & col_mask[None, :]
                new_canvas = jnp.where(crop_mask, self.EMPTY_CELL, canvas)
                return new_canvas, cursor

            # Move to origin: shift grid so cursor becomes (0,0)
            def move_to_origin():
                row_offset, col_offset = cursor[0], cursor[1]
                new_canvas = shift_grid_to_origin(
                    canvas, row_offset, col_offset, 
                    self.GRID_SIZE, self.EMPTY_CELL, self.BACKGROUND_COLOR
                )
                new_cursor = jnp.array([0, 0], dtype=jnp.int32)
                return new_canvas, new_cursor

            branches = [
                act_up,          # 0
                act_down,        # 1
                act_left,        # 2
                act_right,       # 3
                paint,           # 4 – colour 0
                paint,           # 5
                paint,           # 6
                paint,           # 7
                paint,           # 8
                paint,           # 9
                paint,           # 10
                paint,           # 11
                paint,           # 12
                paint,           # 13 – colour 9
                copy,            # 14
                erase,           # 15
                crop,            # 16
                move_to_origin,  # 17
            ]
            return jax.lax.switch(a, branches)

        new_canvas, new_cursor = _apply(action, state.canvas, state.cursor)

        new_steps = state.steps + 1

        # Compute valid mask (where target is not the padding sentinel)
        valid_mask = compute_valid_mask(state.target, self.EMPTY_CELL)
        
        # Reward: incremental improvement only on valid (non-padded) cells
        prev_matches = (state.canvas == state.target) & valid_mask
        new_matches = (new_canvas == state.target) & valid_mask
        prev_correct = jnp.sum(prev_matches)
        new_correct = jnp.sum(new_matches)
        reward = (new_correct - prev_correct).astype(jnp.float32)

        # Done when step budget is exhausted or valid cells match target
        # (solved condition only checks valid cells)
        valid_cells_match = jnp.all((new_canvas == state.target) | ~valid_mask)
        new_done = jnp.logical_or(new_steps >= self.max_steps, valid_cells_match)

        next_state = ARCEnvState(
            rng=state.rng,
            canvas=new_canvas,
            cursor=new_cursor,
            inp=state.inp,
            target=state.target,
            done=new_done,
            steps=new_steps,
            episode_idx=state.episode_idx,
        )

        return next_state, reward, new_done

    # ------------------------------------------------------------------
    # Rendering (debugging – not JIT-friendly)
    # ------------------------------------------------------------------
    def env_render(self, state: ARCEnvState, mode: str = "canvasOnly"):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        colours = [
            "#000000",  # 0 black
            "#0074D9",  # 1 blue
            "#FF4136",  # 2 red
            "#2ECC40",  # 3 green
            "#FFDC00",  # 4 yellow
            "#AAAAAA",  # 5 grey
            "#F012BE",  # 6 fuchsia
            "#FF851B",  # 7 orange
            "#7FDBFF",  # 8 light blue
            "#870C25",  # 9 burgundy
            "#FFFFFF",  # 10 placeholder for -1 (white)
        ]
        cmap = ListedColormap(colours)

        def _prep(arr):
            arr = np.array(arr, dtype=np.int32)
            return np.where(arr < 0, 10, arr)

        def _imshow(ax, grid, title):
            ax.imshow(_prep(grid), cmap=cmap, vmin=0, vmax=10)
            ax.set_title(title)
            ax.axis("off")

        if mode == "canvasOnly":
            fig, ax = plt.subplots(figsize=(4, 4))
            _imshow(ax, state.canvas, "Canvas")
            plt.show()
            return

        # Otherwise, build a gallery
        plots = []
        if mode in {"all", "train"}:
            plots += [(self.train_input[i], f"Train {i} Input") for i in range(self.train_input.shape[0])]
            plots += [(self.train_output[i], f"Train {i} Target") for i in range(self.train_output.shape[0])]
        if mode in {"all", "test"}:
            plots += [(self.test_input[i], f"Test {i} Input") for i in range(self.test_input.shape[0])]
            plots += [(self.test_output[i], f"Test {i} Target") for i in range(self.test_output.shape[0])]

        plots.append((state.canvas, "Current Canvas"))

        n_cols = 2
        n_rows = (len(plots) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        for ax, (grid, title) in zip(axes, plots):
            _imshow(ax, grid, title)

        for ax in axes[len(plots):]:
            ax.axis("off")

        fig.tight_layout()
        plt.show()