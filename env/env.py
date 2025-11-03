import functools
import json
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .types import ARCEnvState
from .utils import compute_valid_mask, pad_to_30, flood_fill_n4, shift_grid_to_origin

__all__ = ["ARCEnv"]


class ARCEnv:
    """A JAX-compatible environment tailored for ARC grid tasks.

    Actions:
        0: move cursor up
        1: move cursor down
        2: move cursor left
        3: move cursor right
        4-13: choose paint color c = (action - 4) in [0..9]
        14: paint selected color at cursor
        15: flood fill from cursor with selected color (4-neighborhood)
        16: crop – fill everything right and down from cursor with -1
        17: move-to-origin – shift grid so cursor moves to (0,0)
        18: send – terminate episode and evaluate reward
        19: copy the entire input grid onto the canvas

    Notes:
        - Default max steps is 1024; on timeout, the episode auto-terminates
          and the terminal reward is evaluated.
        - Reward supports two modes: 'sparse' (default) and 'dense'.
    """

    GRID_SIZE: int = 30
    DEFAULT_MAX_STEPS: int = 1024
    NUM_ACTIONS: int = 20

    EMPTY_CELL: int = -1
    BACKGROUND_COLOR: int = 0

    ACT_UP: int = 0
    ACT_DOWN: int = 1
    ACT_LEFT: int = 2
    ACT_RIGHT: int = 3
    ACT_CHOOSE_COLOR_START: int = 4
    ACT_CHOOSE_COLOR_END: int = 13
    ACT_PAINT: int = 14
    ACT_FLOOD_FILL: int = 15
    ACT_CROP: int = 16
    ACT_MOVE_TO_ORIGIN: int = 17
    ACT_SEND: int = 18
    ACT_COPY: int = 19

    def __init__(
        self,
        train_input: jnp.ndarray,
        train_output: jnp.ndarray,
        test_input: jnp.ndarray,
        test_output: jnp.ndarray,
        max_steps: int = DEFAULT_MAX_STEPS,
        reward_mode: str = "sparse",
        *,
        train_group_ptrs: Optional[jnp.ndarray] = None,
        train_group_sizes: Optional[jnp.ndarray] = None,
        train_pair_problem_ids: Optional[jnp.ndarray] = None,
        test_pair_problem_ids: Optional[jnp.ndarray] = None,
        max_demo_pairs: int = 5,
    ):
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output
        self.max_steps = int(max_steps)
        self.dense_reward = reward_mode == "dense"
        self.max_demo_pairs = max(1, int(max_demo_pairs))

        train_pair_count = int(train_input.shape[0])
        test_pair_count = int(test_input.shape[0])

        if train_pair_problem_ids is None:
            if train_pair_count == 0:
                train_pair_problem_ids = jnp.zeros((0,), dtype=jnp.int32)
            else:
                train_pair_problem_ids = jnp.arange(train_pair_count, dtype=jnp.int32)
        else:
            train_pair_problem_ids = train_pair_problem_ids.astype(jnp.int32)

        if test_pair_problem_ids is None:
            if test_pair_count == 0:
                test_pair_problem_ids = jnp.zeros((0,), dtype=jnp.int32)
            elif train_pair_count > 0:
                test_pair_problem_ids = jnp.zeros((test_pair_count,), dtype=jnp.int32)
            else:
                test_pair_problem_ids = jnp.arange(test_pair_count, dtype=jnp.int32)
        else:
            test_pair_problem_ids = test_pair_problem_ids.astype(jnp.int32)

        self.train_pair_problem_ids = train_pair_problem_ids
        self.test_pair_problem_ids = test_pair_problem_ids

        if train_group_ptrs is None or train_group_sizes is None:
            if train_pair_count == 0:
                train_group_ptrs = jnp.array([0], dtype=jnp.int32)
                train_group_sizes = jnp.zeros((0,), dtype=jnp.int32)
            else:
                ptrs = np.arange(train_pair_count + 1, dtype=np.int32)
                train_group_ptrs = jnp.asarray(ptrs)
                train_group_sizes = jnp.ones((train_pair_count,), dtype=jnp.int32)
        else:
            train_group_ptrs = train_group_ptrs.astype(jnp.int32)
            train_group_sizes = train_group_sizes.astype(jnp.int32)

        self.train_group_ptrs = train_group_ptrs
        self.train_group_sizes = train_group_sizes

        num_groups = int(train_group_ptrs.shape[0]) - 1
        demo_indices = np.full((num_groups, self.max_demo_pairs), -1, dtype=np.int32)
        for gid in range(num_groups):
            start = int(train_group_ptrs[gid])
            size = int(train_group_ptrs[gid + 1] - train_group_ptrs[gid])
            take = min(size, self.max_demo_pairs)
            if take > 0:
                demo_indices[gid, :take] = np.arange(start, start + take, dtype=np.int32)
        self.train_demo_indices = jnp.asarray(demo_indices, dtype=jnp.int32)
        self.train_demo_valid = self.train_demo_indices >= 0

        dummy_grid = jnp.full((1, self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32)
        self.demo_dummy_index = jnp.array(train_pair_count, dtype=jnp.int32)
        if train_pair_count == 0:
            self.train_input_with_dummy = dummy_grid
            self.train_output_with_dummy = dummy_grid
        else:
            self.train_input_with_dummy = jnp.concatenate([train_input, dummy_grid], axis=0)
            self.train_output_with_dummy = jnp.concatenate([train_output, dummy_grid], axis=0)

    @staticmethod
    def from_json(json_str: str, max_steps: int = DEFAULT_MAX_STEPS, reward_mode: str = "sparse") -> "ARCEnv":
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
        train_size = tr_in.shape[0]
        if train_size > 0:
            train_group_ptrs = jnp.array([0, train_size], dtype=jnp.int32)
            train_group_sizes = jnp.array([train_size], dtype=jnp.int32)
            train_problem_ids = jnp.zeros((train_size,), dtype=jnp.int32)
        else:
            train_group_ptrs = jnp.array([0], dtype=jnp.int32)
            train_group_sizes = jnp.zeros((0,), dtype=jnp.int32)
            train_problem_ids = jnp.zeros((0,), dtype=jnp.int32)
        test_problem_ids = jnp.zeros((te_in.shape[0],), dtype=jnp.int32)
        return ARCEnv(
            tr_in,
            tr_out,
            te_in,
            te_out,
            max_steps,
            reward_mode,
            train_group_ptrs=train_group_ptrs,
            train_group_sizes=train_group_sizes,
            train_pair_problem_ids=train_problem_ids,
            test_pair_problem_ids=test_problem_ids,
        )

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def env_reset(self, rng: jax.Array, train: bool = True) -> ARCEnvState:
        """Sample a random task and return an initial State object."""
        rng_next, idx_rng = jax.random.split(rng)

        if train:
            inputs = self.train_input
            targets = self.train_output
            problem_ids = self.train_pair_problem_ids
            split_name = "train"
        else:
            inputs = self.test_input
            targets = self.test_output
            problem_ids = self.test_pair_problem_ids
            split_name = "test"

        n = inputs.shape[0]
        if n == 0:
            raise ValueError(f"{split_name.capitalize()} dataset is empty. Cannot reset environment.")
        idx = jax.random.randint(idx_rng, shape=(), minval=0, maxval=n, dtype=jnp.int32)
        inp = inputs[idx]
        target = targets[idx]
        problem_idx = jnp.asarray(problem_ids[idx] if problem_ids.size > 0 else 0, dtype=jnp.int32)

        canvas = jnp.full((self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32)
        cursor = jnp.array([0, 0], dtype=jnp.int32)
        valid_mask = compute_valid_mask(target, self.EMPTY_CELL)
        baseline_score = jnp.array(0.0, dtype=jnp.float32)
        initial_progress = self._compute_score(canvas, target, valid_mask)

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
            problem_idx=problem_idx,
            selected_color=jnp.array(0, dtype=jnp.int32),
            last_action=jnp.array(-1, dtype=jnp.int32),
            baseline_score=baseline_score,
            prev_progress=initial_progress,
        )

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def env_reset_with_idx(self, rng: jax.Array, idx: jnp.ndarray, train: bool = True) -> ARCEnvState:
        """Deterministic reset to a given task index (useful for vmaps/tests)."""
        rng_next = jax.random.split(rng, 2)[0]
        if train:
            inputs = self.train_input
            targets = self.train_output
            problem_ids = self.train_pair_problem_ids
            split_name = "train"
        else:
            inputs = self.test_input
            targets = self.test_output
            problem_ids = self.test_pair_problem_ids
            split_name = "test"

        n = inputs.shape[0]
        if n == 0:
            raise ValueError(f"{split_name.capitalize()} dataset is empty. Cannot reset environment.")
        inp = inputs[idx]
        target = targets[idx]
        problem_idx = jnp.asarray(problem_ids[idx] if problem_ids.size > 0 else 0, dtype=jnp.int32)
        canvas = jnp.full((self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32)
        cursor = jnp.array([0, 0], dtype=jnp.int32)
        valid_mask = compute_valid_mask(target, self.EMPTY_CELL)
        baseline_score = jnp.array(0.0, dtype=jnp.float32)
        initial_progress = self._compute_score(canvas, target, valid_mask)

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
            problem_idx=problem_idx,
            selected_color=jnp.array(0, dtype=jnp.int32),
            last_action=jnp.array(-1, dtype=jnp.int32),
            baseline_score=baseline_score,
            prev_progress=initial_progress,
        )

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def env_reset_batch(self, rng: jax.Array, train: bool = True, batch_size: int = 1) -> ARCEnvState:
        """Vectorized reset that samples a batch of tasks in one go."""
        keys = jax.random.split(rng, batch_size + 1)
        per_env_rng = keys[1:]

        if train:
            inputs = self.train_input
            targets = self.train_output
            problem_ids = self.train_pair_problem_ids
            split_name = "train"
        else:
            inputs = self.test_input
            targets = self.test_output
            problem_ids = self.test_pair_problem_ids
            split_name = "test"

        n = inputs.shape[0]
        if n == 0:
            raise ValueError(f"{split_name.capitalize()} dataset is empty. Cannot reset environment.")
        idxs = jax.random.randint(keys[0], shape=(batch_size,), minval=0, maxval=n, dtype=jnp.int32)
        inp = jnp.take(inputs, idxs, axis=0)
        target = jnp.take(targets, idxs, axis=0)
        problem_idx = jnp.take(problem_ids, idxs, axis=0) if problem_ids.size > 0 else jnp.zeros_like(idxs)

        canvas = jnp.full((batch_size, self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32)
        cursor = jnp.zeros((batch_size, 2), dtype=jnp.int32)
        valid_mask = compute_valid_mask(target, self.EMPTY_CELL)
        baseline_score = jnp.array(0.0, dtype=jnp.float32)
        initial_progress = self._compute_score(canvas, target, valid_mask)

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
            problem_idx=problem_idx,
            selected_color=jnp.zeros((batch_size,), dtype=jnp.int32),
            last_action=jnp.full((batch_size,), -1, dtype=jnp.int32),
            baseline_score=baseline_score,
            prev_progress=initial_progress,
        )

    def _compute_score(self, canvas: jnp.ndarray, target: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
        canvas_mask = canvas != self.EMPTY_CELL
        valid_mask_bool = valid_mask.astype(bool)
        intersection = jnp.sum((canvas_mask & valid_mask_bool).astype(jnp.float32), axis=(-2, -1))
        union = jnp.sum((canvas_mask | valid_mask_bool).astype(jnp.float32), axis=(-2, -1))
        iou = jnp.where(union > 0.0, intersection / union, jnp.ones_like(union))

        valid_cells = jnp.sum(valid_mask_bool.astype(jnp.float32), axis=(-2, -1))
        correct = jnp.sum(((canvas == target) & valid_mask_bool).astype(jnp.float32), axis=(-2, -1))
        valid_accuracy = jnp.where(
            valid_cells > 0.0,
            correct / jnp.maximum(valid_cells, 1.0),
            jnp.ones_like(valid_cells),
        )

        full_match = jnp.all(canvas == target, axis=(-2, -1)).astype(jnp.float32)
        score = 0.2 * iou + valid_accuracy + full_match
        return score.astype(jnp.float32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def build_context_batch(self, problem_indices: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        problem_indices = jnp.asarray(problem_indices, dtype=jnp.int32)
        batch_size = problem_indices.shape[0]
        num_grids = self.max_demo_pairs * 2
        token_dim = self.GRID_SIZE * self.GRID_SIZE

        num_groups = int(self.train_demo_indices.shape[0])
        grid_id_inputs = jnp.arange(self.max_demo_pairs, dtype=jnp.int32)
        grid_id_outputs = grid_id_inputs + self.max_demo_pairs
        grid_id_template = jnp.stack([grid_id_inputs, grid_id_outputs], axis=1).reshape(-1)

        if num_groups == 0:
            empty_grid = jnp.full((self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32)
            task_grids = jnp.broadcast_to(empty_grid, (batch_size, num_grids, self.GRID_SIZE, self.GRID_SIZE))
            grid_ids = jnp.broadcast_to(grid_id_template[None, :], (batch_size, num_grids))
            mask = jnp.zeros((batch_size, 1, 1, num_grids * token_dim), dtype=bool)
            return task_grids, grid_ids, mask

        safe_idx = jnp.clip(problem_indices, 0, num_groups - 1)
        gathered_idx = self.train_demo_indices[safe_idx]
        valid_pairs = self.train_demo_valid[safe_idx]
        gather_idx = jnp.where(valid_pairs, gathered_idx, self.demo_dummy_index)

        demo_inputs = jnp.take(self.train_input_with_dummy, gather_idx, axis=0)
        demo_outputs = jnp.take(self.train_output_with_dummy, gather_idx, axis=0)

        stacked = jnp.stack([demo_inputs, demo_outputs], axis=2)
        task_grids = stacked.reshape(batch_size, num_grids, self.GRID_SIZE, self.GRID_SIZE)

        grid_ids = jnp.broadcast_to(grid_id_template[None, :], (batch_size, num_grids))
        pair_mask = jnp.repeat(valid_pairs, 2, axis=1)
        token_mask = jnp.repeat(pair_mask[..., None], token_dim, axis=2)
        token_mask = token_mask.reshape(batch_size, 1, 1, num_grids * token_dim)

        return task_grids, grid_ids, token_mask.astype(bool)

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

            choose_color_mask = (a >= self.ACT_CHOOSE_COLOR_START) & (a <= self.ACT_CHOOSE_COLOR_END)
            chosen_color = jnp.clip(a - self.ACT_CHOOSE_COLOR_START, 0, self.ACT_CHOOSE_COLOR_END - self.ACT_CHOOSE_COLOR_START)
            selected_color_after = jnp.where(choose_color_mask, chosen_color, s.selected_color)

            paint_mask = a == self.ACT_PAINT
            flood_mask = a == self.ACT_FLOOD_FILL
            crop_mask = a == self.ACT_CROP
            move_origin_mask = a == self.ACT_MOVE_TO_ORIGIN
            send_mask = a == self.ACT_SEND
            copy_mask = a == self.ACT_COPY

            base_canvas = jnp.where(copy_mask, s.inp, s.canvas)

            def _do_flood():
                return flood_fill_n4(base_canvas, cursor_after_move, selected_color_after)

            def _no_flood():
                return base_canvas

            after_flood = jax.lax.cond(flood_mask, _do_flood, _no_flood)

            painted = _write_cell(after_flood, cursor_after_move[0], cursor_after_move[1], selected_color_after)
            # Crop region to EMPTY_CELL (-1) from cursor to bottom-right
            rows = jnp.arange(self.GRID_SIZE, dtype=jnp.int32)
            cols = jnp.arange(self.GRID_SIZE, dtype=jnp.int32)
            crop_mask_rows = rows[:, None] > cursor_after_move[0]
            crop_mask_cols = cols[None, :] > cursor_after_move[1]
            crop_region = crop_mask_rows | crop_mask_cols
            cropped_canvas = jnp.where(crop_region, self.EMPTY_CELL, s.canvas)

            # Move the subgrid so that the cursor lands at (0,0), fill leftover with BACKGROUND_COLOR
            origin_canvas = shift_grid_to_origin(
                s.canvas,
                cursor_after_move[0],
                cursor_after_move[1],
                self.GRID_SIZE,
                self.EMPTY_CELL,
                self.BACKGROUND_COLOR,
            )

            new_canvas = jnp.where(paint_mask, painted, after_flood)
            new_canvas = jnp.where(crop_mask, cropped_canvas, new_canvas)
            new_canvas = jnp.where(move_origin_mask, origin_canvas, new_canvas)

            origin_cursor = jnp.array([0, 0], dtype=jnp.int32)
            new_cursor = jnp.where(move_origin_mask, origin_cursor, cursor_after_move)

            new_steps = s.steps + jnp.array(1, dtype=jnp.int32)
            hit_budget = new_steps >= jnp.array(self.max_steps, dtype=jnp.int32)

            score_new = self._compute_score(new_canvas, s.target, s.valid_mask)
            reward = score_new - s.prev_progress

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
                problem_idx=s.problem_idx,
                selected_color=selected_color_after,
                last_action=a,
                baseline_score=s.baseline_score,
                prev_progress=score_new,
            )
            return next_state, reward, next_done

        return jax.lax.cond(state.done, _already_done, _do_step, state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def env_step_batch(self, state: ARCEnvState, action: jnp.ndarray):
        """Vmap over leading batch dimension of (state, action)."""
        step_fn = lambda s, a: ARCEnv.env_step(self, s, a)
        return jax.vmap(step_fn, in_axes=(0, 0), out_axes=(0, 0, 0))(state, action)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: ARCEnvState):
        """Build observation without leaking targets.

        Returns a dict with:
          - canvas: (30,30) int32
          - cursor: (2,) int32
          - cursor_mask: (30,30) int32 with 1 at cursor
          - last_action_one_hot: (NUM_ACTIONS,) float32
          - selected_color: int32 scalar
          - time_remaining: float32 scalar in [0,1], 1 at start, 0 at timeout
        """
        grid_size = self.GRID_SIZE
        # cursor mask
        zero = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        cur_mask = jax.lax.dynamic_update_slice(
            zero,
            jnp.ones((1, 1), dtype=jnp.int32),
            (jnp.clip(state.cursor[0], 0, grid_size - 1), jnp.clip(state.cursor[1], 0, grid_size - 1)),
        )

        # last action one hot (zero if no action yet)
        idx = jnp.clip(state.last_action, 0, self.NUM_ACTIONS - 1)
        one_hot = jax.nn.one_hot(idx, self.NUM_ACTIONS, dtype=jnp.float32)
        one_hot = one_hot * (state.last_action >= 0).astype(jnp.float32)

        time_remaining = jnp.clip(1.0 - (state.steps.astype(jnp.float32) / float(self.max_steps)), 0.0, 1.0)

        return {
            "canvas": state.canvas,
            "cursor": state.cursor,
            "cursor_mask": cur_mask,
            "last_action_one_hot": one_hot,
            "selected_color": state.selected_color,
            "time_remaining": time_remaining,
        }

    def get_demos(self, train: bool = True):
        """Return demos without including any test targets.

        For train=True: returns (train_inputs, train_outputs, test_inputs)
        For train=False: returns (train_inputs, train_outputs, test_inputs)
        (same signature; train flag kept for symmetry with reset)
        """
        return self.train_input, self.train_output, self.test_input

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

    def render_state_to_array(self, state: ARCEnvState, show_cursor: bool = True):
        """Render a state to a numpy array suitable for GIF creation.
        
        Args:
            state: The environment state to render
            show_cursor: Whether to show the cursor position on the canvas
            
        Returns:
            A numpy array containing the rendered image (RGB)
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            from PIL import Image
            import io
        except ImportError as e:
            raise ImportError(f"render_state_to_array requires matplotlib and PIL: {e}")
        
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
        
        # Prepare grids
        def prep(arr):
            arr = np.array(arr, dtype=np.int32)
            return np.where(arr < 0, 10, arr)
        
        # Create figure with 3 subplots: input, canvas, target
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Input
        axes[0].imshow(prep(state.inp), cmap=cmap, vmin=0, vmax=10)
        axes[0].set_title(f"Input (Step {int(state.steps)})")
        axes[0].axis("off")
        
        # Canvas with cursor
        canvas_display = prep(state.canvas)
        axes[1].imshow(canvas_display, cmap=cmap, vmin=0, vmax=10)
        if show_cursor:
            # Draw cursor as red square
            cursor_y, cursor_x = int(state.cursor[0]), int(state.cursor[1])
            axes[1].add_patch(plt.Rectangle((cursor_x-0.5, cursor_y-0.5), 1, 1, 
                                           fill=False, edgecolor='red', linewidth=3))
        axes[1].set_title(f"Canvas (Step {int(state.steps)})")
        axes[1].axis("off")
        
        # Target
        axes[2].imshow(prep(state.target), cmap=cmap, vmin=0, vmax=10)
        axes[2].set_title("Target")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # Convert to array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return np.array(img)
