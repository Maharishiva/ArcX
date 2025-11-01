#!/usr/bin/env python3
"""
Generate simplified ARC datasets where each test example exactly matches a
randomly chosen training example from the same puzzle. This makes the task
trivial and is useful for quick environment smoke-tests.
"""

import argparse
import copy
import json
import random
from pathlib import Path


def _rewrite_tests(data: dict, rng: random.Random) -> dict:
    """Copy random train examples into the test split for intentional leakage."""
    train_examples = data.get("train", [])
    if not train_examples:
        return data

    test_examples = data.get("test", [])
    if not test_examples:
        return data

    rewritten = []
    for _ in test_examples:
        rewritten.append(copy.deepcopy(rng.choice(train_examples)))
    data["test"] = rewritten
    return data


def _process_dir(src: Path, dst: Path, seed: int) -> None:
    rng = random.Random(seed)
    dst.mkdir(parents=True, exist_ok=True)

    for json_path in sorted(src.glob("*.json")):
        data = json.loads(json_path.read_text())
        updated = _rewrite_tests(data, rng)
        dst_path = dst / json_path.name
        dst_path.write_text(json.dumps(updated) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create easy ARC datasets with leaked test examples.")
    parser.add_argument(
        "--train-src",
        type=Path,
        default=Path("data/training"),
        help="Directory containing original ARC training JSON files.",
    )
    parser.add_argument(
        "--eval-src",
        type=Path,
        default=Path("data/evaluation"),
        help="Directory containing original ARC evaluation JSON files.",
    )
    parser.add_argument(
        "--train-dst",
        type=Path,
        default=Path("data/training_simple"),
        help="Directory to write simplified training JSON files.",
    )
    parser.add_argument(
        "--val-dst",
        type=Path,
        default=Path("data/val_simple"),
        help="Directory to write simplified validation JSON files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random sampling of leaked train examples.",
    )
    args = parser.parse_args()

    _process_dir(args.train_src, args.train_dst, args.seed)
    _process_dir(args.eval_src, args.val_dst, args.seed + 1)


if __name__ == "__main__":
    main()

