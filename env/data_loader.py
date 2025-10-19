import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp

from .env import ARCEnv
from .utils import pad_to_30

__all__ = ["load_arc_json_file", "load_arc_dir", "build_env_from_dir"]


def _stack_pairs(examples: List[Dict]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    inputs, outputs = [], []
    for ex in examples:
        inputs.append(pad_to_30(ex["input"]))
        outputs.append(pad_to_30(ex["output"]))
    if len(inputs) == 0:
        # Create empty arrays with correct shape
        return (
            jnp.zeros((0, 30, 30), dtype=jnp.int32),
            jnp.zeros((0, 30, 30), dtype=jnp.int32),
        )
    return jnp.stack(inputs), jnp.stack(outputs)


def load_arc_json_file(path: Path) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    data = json.loads(Path(path).read_text())
    train_in, train_out = _stack_pairs(data.get("train", []))
    test_in, test_out = _stack_pairs(data.get("test", []))
    return train_in, train_out, test_in, test_out


def load_arc_dir(dir_path: Path) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dir_path = Path(dir_path)
    train_inputs: List[jnp.ndarray] = []
    train_outputs: List[jnp.ndarray] = []
    test_inputs: List[jnp.ndarray] = []
    test_outputs: List[jnp.ndarray] = []

    for file in sorted(dir_path.glob("*.json")):
        tr_in, tr_out, te_in, te_out = load_arc_json_file(file)
        if tr_in.shape[0] > 0:
            train_inputs.append(tr_in)
            train_outputs.append(tr_out)
        if te_in.shape[0] > 0:
            test_inputs.append(te_in)
            test_outputs.append(te_out)

    def _concat(parts: List[jnp.ndarray]) -> jnp.ndarray:
        if len(parts) == 0:
            return jnp.zeros((0, 30, 30), dtype=jnp.int32)
        return jnp.concatenate(parts, axis=0)

    return _concat(train_inputs), _concat(train_outputs), _concat(test_inputs), _concat(test_outputs)


def build_env_from_dir(dir_path: Path, max_steps: int = ARCEnv.DEFAULT_MAX_STEPS) -> ARCEnv:
    tr_in, tr_out, te_in, te_out = load_arc_dir(dir_path)
    return ARCEnv(tr_in, tr_out, te_in, te_out, max_steps=max_steps)


