import functools
import json

import jax
import jax.numpy as jnp

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
        train_group_starts: jnp.ndarray = None,
        train_group_sizes: jnp.ndarray = None,
        max_demo_pairs: int = 5,
    ):
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output
        self.max_steps = int(max_steps)
        self.dense_reward = reward_mode == "dense"
        self.max_demo_pairs = int(max_demo_pairs)
        
        # Problem grouping metadata for multi-pair context
        if train_group_starts is None:
            # Default: treat each pair as its own problem
            self.train_group_starts = jnp.arange(train_input.shape[0], dtype=jnp.int32)
            self.train_group_sizes = jnp.ones((train_input.shape[0],), dtype=jnp.int32)
        else:
            self.train_group_starts = train_group_starts
            self.train_group_sizes = train_group_sizes

    def _compute_score(self, canvas: jnp.ndarray, target: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
        """Compute progress score: 0.2*IoU(valid-shape) + valid-accuracy + 1.0*(full-match).
        
        This is a JIT-friendly helper used for both baseline and current score computation.
        """
        cells_equal = canvas == target
        
        # Valid-region accuracy
        valid_correct = jnp.where(valid_mask, cells_equal, True)
        valid_accuracy = jnp.mean(valid_correct.astype(jnp.float32))
        
        # IoU of valid shape
        canvas_mask = canvas != self.EMPTY_CELL
        iou_intersection = jnp.sum((canvas_mask & valid_mask).astype(jnp.int32)).astype(jnp.float32)
        iou_union = jnp.sum((canvas_mask | valid_mask).astype(jnp.int32)).astype(jnp.float32)
        iou = jnp.where(iou_union > 0.0, iou_intersection / iou_union, jnp.array(1.0, dtype=jnp.float32))
        
        # Full match bonus
        full_match = jnp.all(cells_equal)
        full_match_bonus = jnp.where(full_match, jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
        
        return 0.2 * iou + valid_accuracy + full_match_bonus

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
        return ARCEnv(tr_in, tr_out, te_in, te_out, max_steps, reward_mode)

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

        # Compute baseline score from input grid
        baseline_score = self._compute_score(inp, target, valid_mask)
        # Initial canvas is empty, compute its score
        initial_score = self._compute_score(canvas, target, valid_mask)
        # Initial progress is max(initial_score - baseline, 0)
        prev_progress = jnp.maximum(initial_score - baseline_score, jnp.array(0.0, dtype=jnp.float32))

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
            selected_color=jnp.array(0, dtype=jnp.int32),
            last_action=jnp.array(-1, dtype=jnp.int32),
            baseline_score=baseline_score,
            prev_progress=prev_progress,
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

        # Compute baseline score from input grid
        baseline_score = self._compute_score(inp, target, valid_mask)
        # Initial canvas is empty, compute its score
        initial_score = self._compute_score(canvas, target, valid_mask)
        # Initial progress is max(initial_score - baseline, 0)
        prev_progress = jnp.maximum(initial_score - baseline_score, jnp.array(0.0, dtype=jnp.float32))

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
            selected_color=jnp.array(0, dtype=jnp.int32),
            last_action=jnp.array(-1, dtype=jnp.int32),
            baseline_score=baseline_score,
            prev_progress=prev_progress,
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

        # Compute baseline scores from input grids (vectorized)
        baseline_score = jax.vmap(lambda i, t, m: self._compute_score(i, t, m))(inp, target, valid_mask)
        # Initial canvas is empty, compute its score
        initial_score = jax.vmap(lambda c, t, m: self._compute_score(c, t, m))(canvas, target, valid_mask)
        # Initial progress is max(initial_score - baseline, 0)
        prev_progress = jnp.maximum(initial_score - baseline_score, jnp.array(0.0, dtype=jnp.float32))

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
            selected_color=jnp.zeros((batch_size,), dtype=jnp.int32),
            last_action=jnp.full((batch_size,), -1, dtype=jnp.int32),
            baseline_score=baseline_score,
            prev_progress=prev_progress,
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
            crop_mask_rows = rows[:, None] >= cursor_after_move[0]
            crop_mask_cols = cols[None, :] >= cursor_after_move[1]
            crop_region = crop_mask_rows & crop_mask_cols
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

            # Progress-shaped reward: r_t = max(s_t - b, 0) - max(s_{t-1} - b, 0)
            # s_t = 0.2*IoU(valid-shape) + valid-accuracy + 1.0*(full-match)
            current_score = self._compute_score(new_canvas, s.target, s.valid_mask)
            current_progress = jnp.maximum(current_score - s.baseline_score, jnp.array(0.0, dtype=jnp.float32))
            reward = current_progress - s.prev_progress

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
                selected_color=selected_color_after,
                last_action=a,
                baseline_score=s.baseline_score,
                prev_progress=current_progress,
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

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def build_context_batch(
        self, episode_idxs: jnp.ndarray, train: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Build context grids for a batch of episodes with padding and mask.
        
        For each episode index, gather up to max_demo_pairs train IO pairs from that
        episode's problem, interleave them as [in0, out0, in1, out1, ...], pad to
        2*max_demo_pairs grids, and build a token mask marking valid grids.
        
        Args:
            episode_idxs: (B,) int32 array of sampled pair indices
            train: if True, build context from train split; else empty context
        
        Returns:
            task_grids: (B, 2*K, 30, 30) int32 array of demo grids, K=max_demo_pairs
            grid_type_ids: (B, 2*K) int32 array with type id per grid (input: 2*p, output: 2*p+1)
            token_mask: (B, 2*K*30*30) bool array marking valid (non-padded) tokens
        """
        B = episode_idxs.shape[0]
        K = self.max_demo_pairs
        num_grids = 2 * K
        
        def _build_for_one(idx):
            # Find which problem this pair belongs to
            # Binary search to find group: group_starts[g] <= idx < group_starts[g+1]
            num_groups = self.train_group_starts.shape[0]
            group_idx = jnp.searchsorted(self.train_group_starts, idx, side='right') - 1
            group_idx = jnp.clip(group_idx, 0, num_groups - 1)
            
            start = self.train_group_starts[group_idx]
            size = self.train_group_sizes[group_idx]
            actual_pairs = jnp.minimum(size, K)
            
            # Gather up to K pairs from this problem
            pair_indices = start + jnp.arange(K, dtype=jnp.int32)
            pair_indices = jnp.where(jnp.arange(K) < actual_pairs, pair_indices, 0)
            
            inputs_gathered = jnp.take(self.train_input, pair_indices, axis=0)
            outputs_gathered = jnp.take(self.train_output, pair_indices, axis=0)
            
            # Interleave: [in0, out0, in1, out1, ...]
            grids_interleaved = jnp.empty((num_grids, self.GRID_SIZE, self.GRID_SIZE), dtype=jnp.int32)
            grids_interleaved = grids_interleaved.at[0::2].set(inputs_gathered)
            grids_interleaved = grids_interleaved.at[1::2].set(outputs_gathered)
            
            # Build grid type IDs: input=2*p, output=2*p+1
            type_ids = jnp.arange(num_grids, dtype=jnp.int32)
            
            # Build token mask: mark tokens from valid grids as True
            # Each grid has GRID_SIZE*GRID_SIZE tokens
            grid_mask = jnp.arange(num_grids) < (2 * actual_pairs)
            token_mask = jnp.repeat(grid_mask, self.GRID_SIZE * self.GRID_SIZE)
            
            return grids_interleaved, type_ids, token_mask
        
        if train:
            task_grids, grid_type_ids, token_mask = jax.vmap(_build_for_one)(episode_idxs)
        else:
            # Test split: return empty context (all padded)
            task_grids = jnp.full(
                (B, num_grids, self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_CELL, dtype=jnp.int32
            )
            grid_type_ids = jnp.arange(num_grids, dtype=jnp.int32)[None, :].repeat(B, axis=0)
            token_mask = jnp.zeros((B, num_grids * self.GRID_SIZE * self.GRID_SIZE), dtype=bool)
        
        return task_grids, grid_type_ids, token_mask

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
