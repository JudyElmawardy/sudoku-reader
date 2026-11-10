"""Puzzle generator tests."""

from data.puzzle_generator import generate_puzzle, count_solutions, generate_full_grid


def test_full_grid_valid():
    g = generate_full_grid(seed=0)
    assert all(len(set(row)) == 9 for row in g)
    # columns
    for c in range(9):
        assert len({g[r][c] for r in range(9)}) == 9


def test_puzzle_unique():
    p, s = generate_puzzle(seed=1, n_givens=30)
    assert count_solutions(p) == 1
    # solution matches
    for r in range(9):
        for c in range(9):
            if p[r][c] != 0:
                assert p[r][c] == s[r][c]


def test_reproducible():
    p1, s1 = generate_puzzle(seed=99, n_givens=28)
    p2, s2 = generate_puzzle(seed=99, n_givens=28)
    assert p1 == p2
    assert s1 == s2