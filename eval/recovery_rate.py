"""
Exact-solution recovery rate on N puzzles per misread level (0..4).
Also writes a correction log for every puzzle.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.puzzle_generator import generate_puzzle, inject_misreads
from csp.sudoku_csp import solve_sudoku
from search.correction_search import CorrectionSearch


def make_probs(true_grid, read_grid, misread_log):
    probs = {}
    mis = {(r, c): (t, w) for r, c, t, w in misread_log}
    for r in range(9):
        for c in range(9):
            if read_grid[r][c] == 0:
                continue
            read = read_grid[r][c]
            if (r, c) in mis:
                t, w = mis[(r, c)]
                probs[(r, c)] = {w: 0.52, t: 0.44}
                for d in range(1, 10):
                    if d not in (t, w):
                        probs[(r, c)][d] = 0.005
            else:
                probs[(r, c)] = {read: 0.90}
                for d in range(1, 10):
                    if d != read:
                        probs[(r, c)][d] = 0.0125
    return probs


def grids_equal(a, b) -> bool:
    return all(a[r][c] == b[r][c] for r in range(9) for c in range(9))


def run_recovery(n_per_level=100, max_misreads=4, base_seed=10000, algorithm="ucs",
                 out_dir=Path("artifacts/recovery")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "correction_log.csv"
    summary = {}

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "puzzle_id", "n_misreads", "seed",
            "cell_r", "cell_c", "read_digit", "new_digit", "cost",
            "nodes_expanded", "solvable", "exact_recovery",
        ])

        for n_mis in range(max_misreads + 1):
            exact_flags = []
            solvable_flags = []
            nodes_list = []

            for i in tqdm(range(n_per_level), desc=f"misreads={n_mis}"):
                seed = base_seed + n_mis * 1000 + i
                puzzle, solution = generate_puzzle(seed=seed, n_givens=30)
                corrupted, mis_log = inject_misreads(puzzle, n_misreads=n_mis, seed=seed)
                probs = make_probs(puzzle, corrupted, mis_log)
                read_at = {(r, c): corrupted[r][c] for r in range(9) for c in range(9)}

                searcher = CorrectionSearch(max_corrections=max(1, n_mis + 1), top_k_alternatives=4)
                corrected, corrs, stats = searcher.search(corrupted, probs, algorithm=algorithm)

                solvable = False
                exact = False
                if corrected is not None:
                    sol, _ = solve_sudoku(corrected, use_fc=True, use_ac3=True)
                    solvable = sol is not None
                    if sol is not None:
                        exact = grids_equal(sol, solution)

                exact_flags.append(1.0 if exact else 0.0)
                solvable_flags.append(1.0 if solvable else 0.0)
                nodes_list.append(stats.get("nodes_expanded") or 0)

                if corrs:
                    for (r, c, new_d) in corrs:
                        cost = 0.0
                        if (r, c) in probs:
                            p = probs[(r, c)]
                            cost = max(0.0, p.get(read_at[(r, c)], 0) - p.get(new_d, 0))
                        writer.writerow([
                            i, n_mis, seed, r, c, read_at[(r, c)], new_d,
                            f"{cost:.4f}", stats.get("nodes_expanded"),
                            solvable, exact,
                        ])
                else:
                    writer.writerow([
                        i, n_mis, seed, "", "", "", "", "",
                        stats.get("nodes_expanded"), solvable, exact,
                    ])

            summary[n_mis] = {
                "n": n_per_level,
                "exact_recovery_mean": float(np.mean(exact_flags)),
                "exact_recovery_std": float(np.std(exact_flags)),
                "solvable_mean": float(np.mean(solvable_flags)),
                "solvable_std": float(np.std(solvable_flags)),
                "nodes_mean": float(np.mean(nodes_list)),
                "nodes_std": float(np.std(nodes_list)),
            }
            print(f"  misreads={n_mis}: exact={summary[n_mis]['exact_recovery_mean']:.3f}  "
                  f"solvable={summary[n_mis]['solvable_mean']:.3f}")

    with open(out_dir / "recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    levels = sorted(summary.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(
        levels,
        [summary[m]["exact_recovery_mean"] for m in levels],
        yerr=[summary[m]["exact_recovery_std"] for m in levels],
        fmt="-o", label="exact recovery",
    )
    axes[0].errorbar(
        levels,
        [summary[m]["solvable_mean"] for m in levels],
        yerr=[summary[m]["solvable_std"] for m in levels],
        fmt="-s", label="solvable",
    )
    axes[0].set_xlabel("# injected misreads")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].set_title("Recovery rate vs misreads")

    axes[1].errorbar(
        levels,
        [summary[m]["nodes_mean"] for m in levels],
        yerr=[summary[m]["nodes_std"] for m in levels],
        fmt="-o",
    )
    axes[1].set_xlabel("# injected misreads")
    axes[1].set_ylabel("Nodes expanded")
    axes[1].set_title("Search effort")
    plt.tight_layout()
    plt.savefig(out_dir / "recovery_rate.png", dpi=130)
    plt.close()
    print(f"Saved {out_dir / 'recovery_summary.json'}")
    print(f"Saved {out_dir / 'correction_log.csv'}")
    print(f"Saved {out_dir / 'recovery_rate.png'}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-misreads", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    n = 20 if args.quick else args.n
    run_recovery(n_per_level=n, max_misreads=args.max_misreads)
