"""
The three required experiments with tables, plots, mean ± std over ≥5 seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def exact_match(sol, solution) -> bool:
    if sol is None:
        return False
    return all(sol[r][c] == solution[r][c] for r in range(9) for c in range(9))


def experiment_1(n_puzzles=30, seeds=(0, 1, 2, 3, 4), max_misreads=4):
    algos = ("ucs", "greedy", "astar")
    nodes = {a: {m: [] for m in range(max_misreads + 1)} for a in algos}
    exact = {a: {m: [] for m in range(max_misreads + 1)} for a in algos}
    solvable = {a: {m: [] for m in range(max_misreads + 1)} for a in algos}

    for seed in seeds:
        for i in range(n_puzzles):
            puzzle, solution = generate_puzzle(seed=1000 + seed * 200 + i, n_givens=30)
            for n_mis in range(max_misreads + 1):
                corrupted, log = inject_misreads(puzzle, n_misreads=n_mis, seed=2000 + seed * 200 + i)
                probs = make_probs(puzzle, corrupted, log)
                for algo in algos:
                    searcher = CorrectionSearch(max_corrections=max(1, n_mis + 1))
                    grid, corrs, stats = searcher.search(corrupted, probs, algorithm=algo)
                    nodes[algo][n_mis].append(stats.get("nodes_expanded") or 0)
                    sol = None
                    if grid is not None:
                        sol, _ = solve_sudoku(grid, use_fc=True, use_ac3=True)
                    solvable[algo][n_mis].append(1.0 if sol is not None else 0.0)
                    exact[algo][n_mis].append(1.0 if exact_match(sol, solution) else 0.0)

    summary = {}
    for algo in algos:
        summary[algo] = {}
        for m in range(max_misreads + 1):
            summary[algo][str(m)] = {
                "nodes_mean": float(np.mean(nodes[algo][m])),
                "nodes_std": float(np.std(nodes[algo][m])),
                "solvable_mean": float(np.mean(solvable[algo][m])),
                "solvable_std": float(np.std(solvable[algo][m])),
                "exact_mean": float(np.mean(exact[algo][m])),
                "exact_std": float(np.std(exact[algo][m])),
                "n": len(nodes[algo][m]),
            }
    return summary


def plot_exp1(summary, out_path: Path):
    misreads = sorted(int(m) for m in next(iter(summary.values())).keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    styles = ["-o", "-s", "-^"]
    for algo, style in zip(summary, styles):
        axes[0].errorbar(misreads, [summary[algo][str(m)]["nodes_mean"] for m in misreads],
                         yerr=[summary[algo][str(m)]["nodes_std"] for m in misreads],
                         fmt=style, label=algo)
        axes[1].errorbar(misreads, [summary[algo][str(m)]["solvable_mean"] for m in misreads],
                         yerr=[summary[algo][str(m)]["solvable_std"] for m in misreads],
                         fmt=style, label=algo)
        axes[2].errorbar(misreads, [summary[algo][str(m)]["exact_mean"] for m in misreads],
                         yerr=[summary[algo][str(m)]["exact_std"] for m in misreads],
                         fmt=style, label=algo)
    axes[0].set_xlabel("# misreads"); axes[0].set_ylabel("Nodes"); axes[0].set_title("Search effort"); axes[0].legend()
    axes[1].set_xlabel("# misreads"); axes[1].set_ylabel("Solvable rate"); axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Solvable after correction"); axes[1].legend()
    axes[2].set_xlabel("# misreads"); axes[2].set_ylabel("Exact recovery"); axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Exact solution recovery"); axes[2].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def experiment_2(n_puzzles=30, seeds=(0, 1, 2, 3, 4)):
    configs = [("BT", False, False), ("BT+FC", True, False), ("BT+FC+AC3", True, True)]
    givens_list = [25, 30, 35]
    raw = {name: {g: {"backtracks": [], "assignments": []} for g in givens_list} for name, _, _ in configs}

    for seed in seeds:
        for i in range(n_puzzles):
            for n_givens in givens_list:
                puzzle, _ = generate_puzzle(seed=3000 + seed * 100 + i * 3 + n_givens, n_givens=n_givens)
                for name, fc, ac3 in configs:
                    _, stats = solve_sudoku(puzzle, use_fc=fc, use_ac3=ac3)
                    raw[name][n_givens]["backtracks"].append(stats["backtracks"])
                    raw[name][n_givens]["assignments"].append(stats["assignments"])

    out = {}
    for name in raw:
        out[name] = {}
        for g in givens_list:
            bt = raw[name][g]["backtracks"]
            asg = raw[name][g]["assignments"]
            out[name][str(g)] = {
                "backtracks_mean": float(np.mean(bt)),
                "backtracks_std": float(np.std(bt)),
                "assignments_mean": float(np.mean(asg)),
                "assignments_std": float(np.std(asg)),
                "n": len(bt),
            }
    return out


def plot_exp2(summary, out_path: Path):
    givens = sorted(int(g) for g in next(iter(summary.values())).keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    styles = ["-o", "-s", "-^"]
    for name, style in zip(summary, styles):
        axes[0].errorbar(givens, [summary[name][str(g)]["backtracks_mean"] for g in givens],
                         yerr=[summary[name][str(g)]["backtracks_std"] for g in givens],
                         fmt=style, label=name)
        axes[1].errorbar(givens, [summary[name][str(g)]["assignments_mean"] for g in givens],
                         yerr=[summary[name][str(g)]["assignments_std"] for g in givens],
                         fmt=style, label=name)
    axes[0].set_xlabel("# givens"); axes[0].set_ylabel("Backtracks"); axes[0].set_title("CSP backtracks")
    axes[0].legend(); axes[0].set_yscale("log")
    axes[1].set_xlabel("# givens"); axes[1].set_ylabel("Assignments"); axes[1].set_title("CSP assignments")
    axes[1].legend(); axes[1].set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def experiment_3(n_puzzles=30, seeds=(0, 1, 2, 3, 4), n_misreads=2):
    methods = ["nn_guided", "none", "random"]
    solvable = {m: [] for m in methods}
    exact = {m: [] for m in methods}
    rng = np.random.default_rng(42)

    for seed in seeds:
        for i in range(n_puzzles):
            puzzle, solution = generate_puzzle(seed=4000 + seed * 100 + i, n_givens=30)
            corrupted, log = inject_misreads(puzzle, n_misreads=n_misreads, seed=5000 + seed * 100 + i)
            probs = make_probs(puzzle, corrupted, log)

            searcher = CorrectionSearch(max_corrections=n_misreads + 1)
            grid, _, _ = searcher.search(corrupted, probs, algorithm="ucs")
            sol = None
            if grid is not None:
                sol, _ = solve_sudoku(grid, use_fc=True, use_ac3=True)
            solvable["nn_guided"].append(1.0 if sol is not None else 0.0)
            exact["nn_guided"].append(1.0 if exact_match(sol, solution) else 0.0)

            sol_none, _ = solve_sudoku(corrupted, use_fc=True, use_ac3=True)
            solvable["none"].append(1.0 if sol_none is not None else 0.0)
            exact["none"].append(1.0 if exact_match(sol_none, solution) else 0.0)

            rand_grid = deepcopy(corrupted)
            for r, c, _, _ in log:
                rand_grid[r][c] = int(rng.integers(1, 10))
            sol_rand, _ = solve_sudoku(rand_grid, use_fc=True, use_ac3=True)
            solvable["random"].append(1.0 if sol_rand is not None else 0.0)
            exact["random"].append(1.0 if exact_match(sol_rand, solution) else 0.0)

    summary = {}
    for m in methods:
        summary[m] = {
            "solvable_mean": float(np.mean(solvable[m])),
            "solvable_std": float(np.std(solvable[m])),
            "exact_mean": float(np.mean(exact[m])),
            "exact_std": float(np.std(exact[m])),
            "n": len(solvable[m]),
        }
    return summary


def plot_exp3(summary, out_path: Path):
    methods = list(summary.keys())
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, [summary[m]["solvable_mean"] for m in methods], width,
           yerr=[summary[m]["solvable_std"] for m in methods], label="solvable", capsize=3)
    ax.bar(x + width / 2, [summary[m]["exact_mean"] for m in methods], width,
           yerr=[summary[m]["exact_std"] for m in methods], label="exact recovery", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title("Correction strategy comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="artifacts/experiments")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 8 if args.quick else 30
    seeds = (0, 1, 2, 3, 4)

    print("=== Experiment 1: UCS vs Greedy vs A* ===")
    s1 = experiment_1(n_puzzles=n, seeds=seeds)
    with open(out / "exp1_search.json", "w") as f:
        json.dump(s1, f, indent=2)
    plot_exp1(s1, out / "exp1_search.png")
    print(json.dumps(s1, indent=2))

    print("\n=== Experiment 2: CSP variants ===")
    s2 = experiment_2(n_puzzles=n, seeds=seeds)
    with open(out / "exp2_csp.json", "w") as f:
        json.dump(s2, f, indent=2)
    plot_exp2(s2, out / "exp2_csp.png")
    print(json.dumps(s2, indent=2))

    print("\n=== Experiment 3: Correction strategies ===")
    s3 = experiment_3(n_puzzles=n, seeds=seeds)
    with open(out / "exp3_correction.json", "w") as f:
        json.dump(s3, f, indent=2)
    plot_exp3(s3, out / "exp3_correction.png")
    print(json.dumps(s3, indent=2))

    print(f"\nAll experiment outputs -> {out}")


if __name__ == "__main__":
    main()
