"""CSP unit tests: normal, edge, impossible cases."""

import pytest
from csp.sudoku_csp import SudokuCSP, solve_sudoku


EASY = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


def test_easy_solvable_bt():
    sol, stats = solve_sudoku(EASY, use_fc=False, use_ac3=False)
    assert sol is not None
    assert stats["solved"]


def test_easy_solvable_fc():
    sol, stats = solve_sudoku(EASY, use_fc=True, use_ac3=False)
    assert sol is not None


def test_easy_solvable_ac3():
    sol, stats = solve_sudoku(EASY, use_fc=True, use_ac3=True)
    assert sol is not None


def test_already_complete():
    # take a known solution
    sol, _ = solve_sudoku(EASY)
    sol2, stats = solve_sudoku(sol)
    assert sol2 == sol
    assert stats["assignments"] == 0


def test_empty_grid_solvable():
    empty = [[0] * 9 for _ in range(9)]
    sol, stats = solve_sudoku(empty, use_fc=True, use_ac3=True)
    assert sol is not None


# ---- edge cases ----

def test_single_given():
    g = [[0] * 9 for _ in range(9)]
    g[0][0] = 5
    sol, _ = solve_sudoku(g)
    assert sol is not None
    assert sol[0][0] == 5


def test_almost_full():
    sol, _ = solve_sudoku(EASY)
    # remove one cell
    puzzle = [row[:] for row in sol]
    puzzle[4][4] = 0
    recovered, _ = solve_sudoku(puzzle)
    assert recovered == sol


def test_counters_increase():
    solver = SudokuCSP(use_fc=False, use_ac3=False)
    solver.solve(EASY)
    assert solver.assignments > 0


# ---- impossible cases ----

def test_duplicate_in_row():
    g = [row[:] for row in EASY]
    g[0][2] = 5  # two 5s in first row
    sol, stats = solve_sudoku(g)
    assert sol is None
    assert not stats["solved"]


def test_duplicate_in_box():
    g = [[0] * 9 for _ in range(9)]
    g[0][0] = 1
    g[1][1] = 1  # same box
    sol, _ = solve_sudoku(g)
    assert sol is None


def test_no_solution_message_style():
    """Impossible cases must give a clear 'no solution' (None)."""
    g = [[1, 1] + [0] * 7] + [[0] * 9 for _ in range(8)]
    sol, stats = solve_sudoku(g)
    assert sol is None
    assert stats["solved"] is False