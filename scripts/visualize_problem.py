#!/usr/bin/env python3
"""Quick visualization script to display all grids from a random training problem."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ARC color palette
COLOURS = [
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
    "#FFFFFF",  # 10 white (for -1 empty cells)
]
CMAP = ListedColormap(COLOURS)


def pad_to_30(grid_list):
    """Pad a grid to 30x30 with -1."""
    grid = np.array(grid_list, dtype=np.int32)
    h, w = grid.shape
    if h >= 30 and w >= 30:
        return grid[:30, :30]
    padded = np.full((30, 30), -1, dtype=np.int32)
    padded[:min(h, 30), :min(w, 30)] = grid[:min(h, 30), :min(w, 30)]
    return padded


def prep_grid(arr):
    """Convert grid to displayable format (map -1 to 10)."""
    arr = np.array(arr, dtype=np.int32)
    return np.where(arr < 0, 10, arr)


def load_problem_from_file(json_path: Path):
    """Load a single problem JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    train_pairs = data.get("train", [])
    test_pairs = data.get("test", [])
    
    train_inputs = [pad_to_30(pair["input"]) for pair in train_pairs]
    train_outputs = [pad_to_30(pair["output"]) for pair in train_pairs]
    test_inputs = [pad_to_30(pair["input"]) for pair in test_pairs]
    test_outputs = [pad_to_30(pair["output"]) for pair in test_pairs]
    
    return train_inputs, train_outputs, test_inputs, test_outputs


def visualize_problem_from_file(json_path: Path, output_path: Path):
    """Visualize all train and test pairs from a JSON file."""
    train_inputs, train_outputs, test_inputs, test_outputs = load_problem_from_file(json_path)
    
    n_train = len(train_inputs)
    n_test = len(test_inputs)
    
    if n_train == 0 and n_test == 0:
        print(f"No pairs found in {json_path.name}")
        return
    
    # Calculate grid layout
    total_pairs = n_train + n_test
    n_cols = 2  # input, output
    n_rows = total_pairs
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    row = 0
    
    # Plot train pairs
    for idx in range(n_train):
        inp = train_inputs[idx]
        out = train_outputs[idx]
        
        axes[row, 0].imshow(prep_grid(inp), cmap=CMAP, vmin=0, vmax=10)
        axes[row, 0].set_title(f"Train {row} Input", fontsize=14, fontweight='bold')
        axes[row, 0].axis("off")
        
        axes[row, 1].imshow(prep_grid(out), cmap=CMAP, vmin=0, vmax=10)
        axes[row, 1].set_title(f"Train {row} Output", fontsize=14, fontweight='bold')
        axes[row, 1].axis("off")
        
        row += 1
    
    # Plot test pairs
    for test_idx in range(n_test):
        inp = test_inputs[test_idx]
        out = test_outputs[test_idx]
        
        axes[row, 0].imshow(prep_grid(inp), cmap=CMAP, vmin=0, vmax=10)
        axes[row, 0].set_title(f"Test {test_idx} Input", fontsize=14, fontweight='bold', color='red')
        axes[row, 0].axis("off")
        
        axes[row, 1].imshow(prep_grid(out), cmap=CMAP, vmin=0, vmax=10)
        axes[row, 1].set_title(f"Test {test_idx} Output", fontsize=14, fontweight='bold', color='red')
        axes[row, 1].axis("off")
        
        row += 1
    
    problem_id = json_path.stem
    fig.suptitle(f"Problem {problem_id} ({n_train} train, {n_test} test)", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    print(f"  Problem: {problem_id}")
    print(f"  Train pairs: {n_train}")
    print(f"  Test pairs: {n_test}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize a random ARC problem")
    parser.add_argument("--data-dir", type=str, default="data/training_simple", 
                        help="Path to ARC data directory")
    parser.add_argument("--problem-id", type=str, default=None,
                        help="Specific problem ID (JSON filename without .json) to visualize")
    parser.add_argument("--output", type=str, default="artifacts/problem_viz.png",
                        help="Output PNG path")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: directory {data_dir} does not exist")
        return
    
    # Get all JSON files
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} problems in {data_dir}")
    
    # Select problem
    if args.problem_id is not None:
        problem_file = data_dir / f"{args.problem_id}.json"
        if not problem_file.exists():
            print(f"Error: problem {args.problem_id}.json not found in {data_dir}")
            return
        selected_file = problem_file
        print(f"Visualizing problem: {args.problem_id}")
    else:
        # Use current time as seed for true randomness
        seed = int(time.time() * 1000) % (2**32)
        random.seed(seed)
        selected_file = random.choice(json_files)
        print(f"Randomly selected problem: {selected_file.stem} (seed: {seed})")
    
    visualize_problem_from_file(selected_file, Path(args.output))


if __name__ == "__main__":
    main()

