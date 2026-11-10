"""
Integrated demo: python run_demo.py

Default mode uses digit-level misreads + synthetic NN probabilities
so recovery is reliable and demonstrable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.puzzle_generator import generate_puzzle, inject_misreads
from csp.sudoku_csp import solve_sudoku
from search.correction_search import CorrectionSearch


def print_grid(grid, title=""):
    if title:
        print(title)
    for r in range(9):
        print("  " + " ".join(str(v) if v else "." for v in grid[r]))
    print()


def make_synthetic_probs(true_grid, read_grid, misread_log):
    probs = {}
    misread_cells = {(r, c): (true, wrong) for r, c, true, wrong in misread_log}
    for r in range(9):
        for c in range(9):
            if read_grid[r][c] == 0:
                continue
            read = read_grid[r][c]
            if (r, c) in misread_cells:
                true, wrong = misread_cells[(r, c)]
                probs[(r, c)] = {wrong: 0.52, true: 0.44}
                for d in range(1, 10):
                    if d not in (true, wrong):
                        probs[(r, c)][d] = 0.005
            else:
                probs[(r, c)] = {read: 0.88}
                for d in range(1, 10):
                    if d != read:
                        probs[(r, c)][d] = 0.015
    return probs


def demo_logic(seed, n_misreads, algorithm="ucs"):
    print("=" * 60)
    print("DEMO: logic mode (digit misreads + synthetic NN probs)")
    print("=" * 60)

    puzzle, solution = generate_puzzle(seed=seed, n_givens=30)
    print_grid(puzzle, "=== True puzzle (givens) ===")

    corrupted, misread_log = inject_misreads(puzzle, n_misreads=n_misreads, seed=seed)
    print("Injected misreads (r, c, true, wrong):", misread_log)
    print_grid(corrupted, "=== Corrupted reading ===")

    sol0, stats0 = solve_sudoku(corrupted, use_fc=True, use_ac3=True)
    print(f"Initially solvable? {sol0 is not None}  CSP stats: {stats0}")

    probs = make_synthetic_probs(puzzle, corrupted, misread_log)

    print(f"\n--- Search ({algorithm}) ---")
    searcher = CorrectionSearch(max_corrections=max(3, n_misreads + 1), top_k_alternatives=4)
    corrected, corrs, search_stats = searcher.search(corrupted, probs, algorithm=algorithm)
    print("Corrections:", corrs)
    print("Search stats:", search_stats)

    if corrected is None:
        print("STATUS: unreadable")
        return False

    print_grid(corrected, "=== Corrected grid ===")
    sol, csp_stats = solve_sudoku(corrected, use_fc=True, use_ac3=True)
    print("CSP stats:", csp_stats)
    if sol is None:
        print("STATUS: unreadable after correction")
        return False

    print_grid(sol, "=== Solution ===")
    exact = all(sol[r][c] == solution[r][c] for r in range(9) for c in range(9))
    print(f"Exact recovery: {exact}")
    print("STATUS: ok")
    return True


def demo_csp_pruning(seed=0):
    print("\n" + "=" * 60)
    print("CSP: BT vs BT+FC vs BT+FC+AC-3")
    print("=" * 60)
    puzzle, _ = generate_puzzle(seed=seed, n_givens=28)
    for name, fc, ac3 in [
        ("BT only", False, False),
        ("BT + FC", True, False),
        ("BT + FC + AC-3", True, True),
    ]:
        sol, stats = solve_sudoku(puzzle, use_fc=fc, use_ac3=ac3)
        print(f"  {name:16s}  solved={sol is not None}  "
              f"assignments={stats['assignments']:5d}  backtracks={stats['backtracks']:5d}")


def demo_nn_on_vs_off(seed, n_misreads):
    print("\n" + "=" * 60)
    print("NN ON vs OFF")
    print("=" * 60)
    puzzle, _ = generate_puzzle(seed=seed, n_givens=30)
    corrupted, log = inject_misreads(puzzle, n_misreads=n_misreads, seed=seed)

    sol_off, _ = solve_sudoku(corrupted, use_fc=True, use_ac3=True)
    print(f"  NN OFF: solvable={sol_off is not None}")

    probs = make_synthetic_probs(puzzle, corrupted, log)
    grid, corrs, stats = CorrectionSearch(max_corrections=3).search(corrupted, probs, "ucs")
    sol_on = None
    if grid is not None:
        sol_on, _ = solve_sudoku(grid, use_fc=True, use_ac3=True)
    print(f"  NN ON : solvable={sol_on is not None}  corrections={corrs}")
    print(f"  nodes expanded: {stats.get('nodes_expanded')}")


def save_plot(seed, n_misreads, out_path):
    puzzle, solution = generate_puzzle(seed=seed, n_givens=30)
    corrupted, log = inject_misreads(puzzle, n_misreads=n_misreads, seed=seed)
    probs = make_synthetic_probs(puzzle, corrupted, log)
    corrected, corrs, _ = CorrectionSearch(max_corrections=3).search(corrupted, probs, "ucs")
    sol, _ = solve_sudoku(corrected) if corrected else (None, None)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def draw(ax, grid, title):
        ax.set_title(title)
        ax.axis("off")
        if grid is None:
            ax.text(0.5, 0.5, "none", ha="center", va="center")
            return
        text = "\n".join(" ".join(str(v) if v else "." for v in row) for row in grid)
        ax.text(0.1, 0.5, text, family="monospace", fontsize=11, va="center")

    draw(axes[0], corrupted, "Reading (corrupted)")
    draw(axes[1], corrected, f"After search ({len(corrs) if corrs else 0} fixes)")
    draw(axes[2], sol, "CSP solution")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--misreads", type=int, default=2)
    parser.add_argument("--algorithm", default="ucs", choices=["ucs", "greedy", "astar"])
    parser.add_argument("--out", type=str, default="figures/demo.png")
    args = parser.parse_args()

    ok = demo_logic(args.seed, args.misreads, args.algorithm)
    demo_csp_pruning(seed=args.seed)
    demo_nn_on_vs_off(args.seed, args.misreads)
    save_plot(args.seed, args.misreads, args.out)

    if not ok:
        print("\n*** Demo failed ***")
        raise SystemExit(1)
    print("\n*** Demo completed successfully ***")


if __name__ == "__main__":
    main()
