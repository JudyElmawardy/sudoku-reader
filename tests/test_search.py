"""Search unit tests + the required small known-answer example."""

from search.correction_search import CorrectionSearch, run_small_example


def test_small_known_example():
    """Checklist item 5: tested on a small example with a known answer."""
    run_small_example()  # asserts internally


def test_already_solvable_returns_zero_cost():
    # trivial empty-ish grid that is solvable
    base = [[0] * 9 for _ in range(9)]
    base[0][0] = 1
    probs = {(0, 0): {1: 0.9, 2: 0.1}}
    searcher = CorrectionSearch(max_corrections=2)
    grid, corrs, stats = searcher.search(base, probs, algorithm="ucs")
    assert grid is not None
    assert corrs == []
    assert stats["cost"] == 0.0


def test_ucs_vs_greedy_both_find_solution():
    base = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    probs = {
        (0, 0): {1: 0.9, 2: 0.05, 3: 0.05},
        (0, 1): {1: 0.55, 2: 0.40, 3: 0.05},
    }
    for algo in ("ucs", "greedy", "astar"):
        searcher = CorrectionSearch(max_corrections=2)
        grid, corrs, stats = searcher.search(base, probs, algorithm=algo)
        assert grid is not None, f"{algo} failed"
        assert stats["nodes_expanded"] >= 0