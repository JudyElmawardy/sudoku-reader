"""
Seeded Sudoku puzzle generator.

Generates a complete unique grid via randomised backtracking, then removes
cells while uniqueness is preserved.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import List, Optional, Tuple

Grid = List[List[int]]


def _find_empty(grid: Grid) -> Optional[Tuple[int, int]]:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None


def _valid(grid: Grid, r: int, c: int, val: int) -> bool:
    if any(grid[r][j] == val for j in range(9)):
        return False
    if any(grid[i][c] == val for i in range(9)):
        return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if grid[i][j] == val:
                return False
    return True


def _fill(grid: Grid) -> bool:
    """Fill the grid in-place with one complete solution. Returns success."""
    empty = _find_empty(grid)
    if empty is None:
        return True
    r, c = empty
    nums = list(range(1, 10))
    random.shuffle(nums)
    for val in nums:
        if _valid(grid, r, c, val):
            grid[r][c] = val
            if _fill(grid):
                return True
            grid[r][c] = 0
    return False


def _count_solutions(grid: Grid, limit: int = 2) -> int:
    """Count solutions up to `limit` (mutates a copy)."""
    empty = _find_empty(grid)
    if empty is None:
        return 1
    r, c = empty
    found = 0
    for val in range(1, 10):
        if _valid(grid, r, c, val):
            grid[r][c] = val
            found += _count_solutions(grid, limit)
            grid[r][c] = 0
            if found >= limit:
                return found
    return found


def generate_full_grid(seed: int) -> Grid:
    """Generate a complete valid Sudoku grid (randomised)."""
    random.seed(seed)
    grid = [[0] * 9 for _ in range(9)]
    # fill diagonal boxes first (they are independent)
    for box in range(3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        idx = 0
        for i in range(3):
            for j in range(3):
                grid[3 * box + i][3 * box + j] = nums[idx]
                idx += 1
    assert _fill(grid), "Failed to complete grid"
    return grid


def count_solutions(grid: Grid, limit: int = 2) -> int:
    g = deepcopy(grid)
    return _count_solutions(g, limit=limit)


def generate_puzzle(
    seed: int,
    n_givens: int = 30,
    max_attempts: int = 80,
) -> Tuple[Grid, Grid]:
    """
    Generate a puzzle with approximately `n_givens` cells that has a unique solution.

    Returns (puzzle, solution).
    """
    random.seed(seed)
    solution = generate_full_grid(seed)
    puzzle = deepcopy(solution)

    positions = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(positions)

    target_remove = 81 - n_givens
    removed = 0
    for r, c in positions:
        if removed >= target_remove:
            break
        saved = puzzle[r][c]
        puzzle[r][c] = 0
        if count_solutions(puzzle, limit=2) != 1:
            puzzle[r][c] = saved
        else:
            removed += 1

    return puzzle, solution


def inject_misreads(
    true_grid: Grid,
    n_misreads: int,
    seed: int,
) -> Tuple[Grid, List[Tuple[int, int, int, int]]]:
    """
    Create a corrupted reading that differs in exactly `n_misreads` non-empty cells.

    Returns (corrupted_grid, list of (r, c, true_digit, wrong_digit)).
    """
    random.seed(seed + 17)
    candidates = [(r, c) for r in range(9) for c in range(9) if true_grid[r][c] != 0]
    random.shuffle(candidates)
    corrupted = deepcopy(true_grid)
    log = []
    for r, c in candidates[:n_misreads]:
        true = true_grid[r][c]
        wrong = random.choice([d for d in range(1, 10) if d != true])
        corrupted[r][c] = wrong
        log.append((r, c, true, wrong))
    return corrupted, log


if __name__ == "__main__":
    p, s = generate_puzzle(seed=42, n_givens=30)
    print("Puzzle givens:", sum(1 for r in p for v in r if v))
    print("Unique?", count_solutions(p) == 1)
    for row in p:
        print(row)