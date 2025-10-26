import functools
import json

import jax
import jax.numpy as jnp

from .types import ARCEnvState
from .utils import compute_valid_mask, pad_to_30, shift_grid_to_origin

__all__ = ["ARCEnv"]


class ARCEnv:
    """A JAX-compatible environment tailored for ARC grid tasks.

    Actions:
        0: move cursor up
        1: move cursor down
        2: move cursor left
        3: move cursor right
        4-13: paint the cell with colour (action-4) ∈ [0-9]
        14: copy the entire input grid onto the canvas
        15: send – terminate episode and evaluate reward
        16: crop – fill everything right and down from cursor with -1
        17: move-to-origin – shift grid so cursor moves to (0,0)
    """

    GRID_SIZE: int = 30
    DEFAULT_MAX_STEPS: int = 100
    NUM_ACTIONS: int = 18

    EMPTY_CELL: int = -1
    BACKGROUND_COLOR: int = 0

    ACT_UP: int = 0
    ACT_DOWN: int = 1
    ACT_LEFT: int = 2
    ACT_RIGHT: int = 3
    ACT_PAINT_START: int = 4
    ACT_PAINT_END: int = 13
    ACT_COPY: int = 14
    ACT_SEND: int = 15
    ACT_CROP: int = 16
    ACT_MOVE_TO_ORIGIN: int = 17

    def __init__(
        self,
        train_input: jnp.ndarray,
        train_output: jnp.ndarray,
        test_input: jnp.ndarray,
        test_output: jnp.ndarray,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output
        self.max_steps = int(max_steps)

    @staticmethod
    def from_json(json_str: str, max_steps: int = DEFAULT_MAX_STEPS) -> "ARCEnv":
        """Create an environment from the raw ARC JSON string."""

        def _load_pairs(examples):
            ins, outs = [], []
            for example in examples:
                ins.append(pad_to_30(example["input"]))
                outs.append(pad_to_30(example["output"]))
            if not ins:
                zeros = jnp.zeros((0, ARCEnv.GRID_SIZE, ARCEnv.GRID_SIZE), dtype=jnp.int32)
                return zeros, zeros
            return jnp.stack(ins), jnp.stack(outs)

        data = json.loads(json_str)
        tr_in, tr_out = _load_pairs(data.get("train", []))
        te_in, te_out = _load_pairs(data.get("test", []))
        return ARCEnv(tr_in, tr_out, te_in, te_out, max_steps)

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def env_reset(self, rng: jax.Array, train: bool = True) -> ARCEnvState:
        """Sample a random task and return an initial State object."""
        rng_next, idx_rng = jax.random.split(rng)

        if train:
            inputs = self.train_input
            targets = self.train_output
            split_name = "train"
        else:
            inputs = self.test_input
            targets = self.test_output
            split_name = "test"

        n = inputs.shape[0]
        if n == 0:
            raise ValueError(f"{split_name.capitalize()} dataset is empty. Cannot reset environment.")
        idx = jax.random.randint(idx_rng, shape=(), minval=0, maxval=n, dtype=jnp.int32)
        inp = inputs[idx]
        target = targets[idx]

        canvas = jnp.full((self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32)
        cursor = jnp.array([0, 0], dtype=jnp.int32)
        valid_mask = compute_valid_mask(target, self.EMPTY_CELL)

        return ARCEnvState(
            rng=rng_next,
            canvas=canvas,
            cursor=cursor,
            inp=inp,
            target=target,
            valid_mask=valid_mask,
            done=jnp.array(False),
            steps=jnp.array(0, dtype=jnp.int32),
            episode_idx=idx,
        )

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def env_reset_with_idx(self, rng: jax.Array, idx: jnp.ndarray, train: bool = True) -> ARCEnvState:
        """Deterministic reset to a given task index (useful for vmaps/tests)."""
        rng_next = jax.random.split(rng, 2)[0]
        if train:
            inputs = self.train_input
            targets = self.train_output
            split_name = "train"
        else:
            inputs = self.test_input
            targets = self.test_output
            split_name = "test"

        n = inputs.shape[0]
        if n == 0:
            raise ValueError(f"{split_name.capitalize()} dataset is empty. Cannot reset environment.")
        inp = inputs[idx]
        target = targets[idx]
        canvas = jnp.full((self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32)
        cursor = jnp.array([0, 0], dtype=jnp.int32)
        valid_mask = compute_valid_mask(target, self.EMPTY_CELL)

        return ARCEnvState(
            rng=rng_next,
            canvas=canvas,
            cursor=cursor,
            inp=inp,
            target=target,
            valid_mask=valid_mask,
            done=jnp.array(False),
            steps=jnp.array(0, dtype=jnp.int32),
            episode_idx=idx,
        )

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def env_reset_batch(self, rng: jax.Array, train: bool = True, batch_size: int = 1) -> ARCEnvState:
        """Vectorized reset that samples a batch of tasks in one go."""
        keys = jax.random.split(rng, batch_size + 1)
        per_env_rng = keys[1:]

        if train:
            inputs = self.train_input
            targets = self.train_output
            split_name = "train"
        else:
            inputs = self.test_input
            targets = self.test_output
            split_name = "test"

        n = inputs.shape[0]
        if n == 0:
            raise ValueError(f"{split_name.capitalize()} dataset is empty. Cannot reset environment.")
        idxs = jax.random.randint(keys[0], shape=(batch_size,), minval=0, maxval=n, dtype=jnp.int32)
        inp = jnp.take(inputs, idxs, axis=0)
        target = jnp.take(targets, idxs, axis=0)

        canvas = jnp.full((batch_size, self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32)
        cursor = jnp.zeros((batch_size, 2), dtype=jnp.int32)
        valid_mask = compute_valid_mask(target, self.EMPTY_CELL)

        return ARCEnvState(
            rng=per_env_rng,
            canvas=canvas,
            cursor=cursor,
            inp=inp,
            target=target,
            valid_mask=valid_mask,
            done=jnp.zeros((batch_size,), dtype=bool),
            steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            episode_idx=idxs,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def env_step(self, state: ARCEnvState, action: jnp.ndarray):
        """One interaction step – returns (next_state, reward, done)."""
        a = jnp.asarray(action, dtype=jnp.int32)
        a = jnp.clip(a, 0, self.NUM_ACTIONS - 1)

        def _already_done(s):
            return s, jnp.array(0.0, dtype=jnp.float32), s.done

        def _write_cell(arr, row, col, value):
            update = jnp.full((1, 1), value, dtype=arr.dtype)
            return jax.lax.dynamic_update_slice(arr, update, (row, col))

        def _do_step(s: ARCEnvState):
            move_mask = a <= self.ACT_RIGHT
            direction_table = jnp.array(
                [[-1, 0], [1, 0], [0, -1], [0, 1]],
                dtype=jnp.int32,
            )
            direction_idx = jnp.clip(a, self.ACT_UP, self.ACT_RIGHT)
            delta = direction_table[direction_idx] * move_mask.astype(jnp.int32)
            cursor_after_move = jnp.clip(s.cursor + delta, 0, self.GRID_SIZE - 1)

            paint_mask = (a >= self.ACT_PAINT_START) & (a <= self.ACT_PAINT_END)
            paint_color = jnp.clip(a - self.ACT_PAINT_START, 0, self.ACT_PAINT_END - self.ACT_PAINT_START)
            paint_canvas = _write_cell(s.canvas, cursor_after_move[0], cursor_after_move[1], paint_color)

            copy_mask = a == self.ACT_COPY
            crop_mask = a == self.ACT_CROP
            move_origin_mask = a == self.ACT_MOVE_TO_ORIGIN
            send_mask = a == self.ACT_SEND

            rows = jnp.arange(self.GRID_SIZE, dtype=jnp.int32)
            cols = jnp.arange(self.GRID_SIZE, dtype=jnp.int32)
            crop_mask_rows = rows[:, None] >= cursor_after_move[0]
            crop_mask_cols = cols[None, :] >= cursor_after_move[1]
            crop_region = crop_mask_rows & crop_mask_cols
            cropped_canvas = jnp.where(crop_region, self.EMPTY_CELL, s.canvas)

            origin_canvas = shift_grid_to_origin(
                s.canvas,
                cursor_after_move[0],
                cursor_after_move[1],
                self.GRID_SIZE,
                self.EMPTY_CELL,
                self.BACKGROUND_COLOR,
            )

            new_canvas = jnp.where(paint_mask, paint_canvas, s.canvas)
            new_canvas = jnp.where(copy_mask, s.inp, new_canvas)
            new_canvas = jnp.where(crop_mask, cropped_canvas, new_canvas)
            new_canvas = jnp.where(move_origin_mask, origin_canvas, new_canvas)

            origin_cursor = jnp.array([0, 0], dtype=jnp.int32)
            new_cursor = jnp.where(move_origin_mask, origin_cursor, cursor_after_move)

            new_steps = s.steps + jnp.array(1, dtype=jnp.int32)
            hit_budget = new_steps >= jnp.array(self.max_steps, dtype=jnp.int32)

            cells_match = new_canvas == s.target
            valid_cells_match = jnp.all(jnp.where(s.valid_mask, cells_match, True))
            send_reward = jnp.where(valid_cells_match, jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
            reward = jnp.where(send_mask, send_reward, jnp.array(0.0, dtype=jnp.float32))

            next_done = jnp.logical_or(send_mask, hit_budget)

            next_state = ARCEnvState(
                rng=s.rng,
                canvas=new_canvas,
                cursor=new_cursor,
                inp=s.inp,
                target=s.target,
                valid_mask=s.valid_mask,
                done=next_done,
                steps=new_steps,
                episode_idx=s.episode_idx,
            )
            return next_state, reward, next_done

        return jax.lax.cond(state.done, _already_done, _do_step, state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def env_step_batch(self, state: ARCEnvState, action: jnp.ndarray):
        """Vmap over leading batch dimension of (state, action)."""
        step_fn = lambda s, a: ARCEnv.env_step(self, s, a)
        return jax.vmap(step_fn, in_axes=(0, 0), out_axes=(0, 0, 0))(state, action)

    def env_render(self, state: ARCEnvState, mode: str = "canvasOnly"):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap

        colours = [
            "#000000",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F012BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",
            "#FFFFFF",
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

        plots = []
        if mode in {"all", "train"}:
            for i in range(self.train_input.shape[0]):
                plots.append((self.train_input[i], f"Train {i} Input"))
                plots.append((self.train_output[i], f"Train {i} Target"))
        if mode in {"all", "test"}:
            for i in range(self.test_input.shape[0]):
                plots.append((self.test_input[i], f"Test {i} Input"))
                plots.append((self.test_output[i], f"Test {i} Target"))

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
