"""
Reproducible data pipeline (fixed seed).

1. Download MNIST
2. Keep only digits 1-9, scale to [0,1]
3. Split original training set → 50k train / 10k val
4. Generate a bank of Sudoku puzzles + mosaics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from data.puzzle_generator import generate_puzzle, inject_misreads


def download_and_preprocess(data_dir: Path, seed: int = 42):
    data_dir.mkdir(parents=True, exist_ok=True)
    transform = T.Compose([T.ToTensor()])  # already [0,1]

    print("Downloading MNIST...")
    train_set = torchvision.datasets.MNIST(
        root=str(data_dir / "raw"), train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=str(data_dir / "raw"), train=False, download=True, transform=transform
    )

    def filter_1_9(dataset):
        xs, ys = [], []
        for img, label in dataset:
            if label == 0:
                continue
            xs.append(img.squeeze(0).numpy())  # (28,28)
            ys.append(label - 1)  # map 1-9 → 0-8
        return np.stack(xs), np.array(ys, dtype=np.int64)

    print("Filtering digits 1-9...")
    X_all, y_all = filter_1_9(train_set)
    X_test, y_test = filter_1_9(test_set)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_all))
    train_idx, val_idx = idx[:50000], idx[50000:60000]
    # original MNIST train has 54k of 1-9 roughly; we take first 50k / next 10k after shuffle
    # if not enough, just split what we have
    n = len(X_all)
    n_train = min(50000, int(0.85 * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    out = data_dir / "mnist_1_9.npz"
    np.savez_compressed(
        out,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print(f"Saved {out}")
    print(f"  train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_puzzle_bank(
    out_dir: Path,
    n_puzzles: int = 200,
    n_givens_list=(25, 30, 35),
    seed: int = 42,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i in tqdm(range(n_puzzles), desc="Generating puzzles"):
        n_givens = n_givens_list[i % len(n_givens_list)]
        puzzle, solution = generate_puzzle(seed=seed + i, n_givens=n_givens)
        path = out_dir / f"puzzle_{i:04d}.npz"
        np.savez_compressed(path, puzzle=np.array(puzzle), solution=np.array(solution))
        meta.append({
            "id": i,
            "path": str(path.name),
            "n_givens": int(sum(1 for r in puzzle for v in r if v)),
            "seed": seed + i,
        })
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Generated {n_puzzles} puzzles → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--n-puzzles", type=int, default=200)
    args = parser.parse_args()

    out = Path(args.out_dir)
    download_and_preprocess(out, seed=args.seed)
    generate_puzzle_bank(out / "puzzles", n_puzzles=args.n_puzzles, seed=args.seed)
    print("Done.")


if __name__ == "__main__":
    main()