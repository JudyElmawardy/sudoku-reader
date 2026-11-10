"""
Sudoku CSP solver with switchable Backtracking, Forward Checking and AC-3.

When use_ac3=True, arc consistency is maintained after every assignment
(MAC – Maintaining Arc Consistency), not only as preprocessing.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

Grid = List[List[int]]
Domain = Dict[Tuple[int, int], Set[int]]
Cell = Tuple[int, int]


class SudokuCSP:
    def __init__(
        self,
        use_fc: bool = True,
        use_ac3: bool = True,
        uniqueness_check: bool = False,
    ):
        self.use_fc = use_fc
        self.use_ac3 = use_ac3
        self.uniqueness_check = uniqueness_check
        self.assignments = 0
        self.backtracks = 0
        self._solution: Optional[Grid] = None
        self._n_solutions = 0

    def reset_counters(self):
        self.assignments = 0
        self.backtracks = 0
        self._solution = None
        self._n_solutions = 0

    def _init_domains(self, grid: Grid) -> Domain:
        domains: Domain = {}
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    domains[(r, c)] = set(range(1, 10))
                else:
                    domains[(r, c)] = {grid[r][c]}
        return domains

    def _peers(self, r: int, c: int) -> List[Cell]:
        peers = set()
        for j in range(9):
            if j != c:
                peers.add((r, j))
        for i in range(9):
            if i != r:
                peers.add((i, c))
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if (i, j) != (r, c):
                    peers.add((i, j))
        return list(peers)

    # ------------------------------------------------------------------
    # Arc Consistency (AC-3) – used both as preprocessing and during search
    # ------------------------------------------------------------------
    def _revise(self, domains: Domain, xi: Cell, xj: Cell) -> bool:
        """Remove values from domains[xi] that have no support in domains[xj]."""
        revised = False
        to_remove = []
        for x in domains[xi]:
            # x is supported if there exists y in domains[xj] with y != x
            if not any(y != x for y in domains[xj]):
                to_remove.append(x)
        for x in to_remove:
            domains[xi].discard(x)
            revised = True
        return revised

    def ac3(self, domains: Domain, queue: Optional[List[Tuple[Cell, Cell]]] = None) -> bool:
        """
        Enforce arc consistency.
        Returns False if some domain becomes empty.
        """
        if queue is None:
            queue = []
            cells = list(domains.keys())
            for xi in cells:
                for xj in self._peers(*xi):
                    if xj in domains:
                        queue.append((xi, xj))

        while queue:
            xi, xj = queue.pop(0)
            if self._revise(domains, xi, xj):
                if not domains[xi]:
                    return False
                for xk in self._peers(*xi):
                    if xk != xj and xk in domains:
                        queue.append((xk, xi))
        return True

    def _ac3_after_assignment(self, domains: Domain, cell: Cell) -> bool:
        """Re-run AC-3 starting from arcs (peer, cell)."""
        queue = []
        for xk in self._peers(*cell):
            if xk in domains:
                queue.append((xk, cell))
        return self.ac3(domains, queue=queue)

    # ------------------------------------------------------------------
    # Forward Checking
    # ------------------------------------------------------------------
    def _forward_check(
        self, domains: Domain, r: int, c: int, val: int
    ) -> Optional[Dict[Cell, Set[int]]]:
        removals: Dict[Cell, Set[int]] = {}
        for pr, pc in self._peers(r, c):
            if (pr, pc) not in domains:
                continue
            if val in domains[(pr, pc)]:
                removals.setdefault((pr, pc), set()).add(val)
                domains[(pr, pc)].discard(val)
                if not domains[(pr, pc)]:
                    # undo
                    for cell, vals in removals.items():
                        domains[cell] |= vals
                    return None
        return removals

    def _undo(self, domains: Domain, removals: Dict[Cell, Set[int]]):
        for cell, vals in removals.items():
            domains[cell] |= vals

    # ------------------------------------------------------------------
    # Variable ordering (MRV)
    # ------------------------------------------------------------------
    def _select_unassigned(self, domains: Domain, grid: Grid) -> Optional[Cell]:
        best = None
        best_size = 10
        for (r, c), dom in domains.items():
            if grid[r][c] == 0 and 0 < len(dom) < best_size:
                best_size = len(dom)
                best = (r, c)
        return best

    # ------------------------------------------------------------------
    # Snapshot / restore domains (for AC-3 during search)
    # ------------------------------------------------------------------
    def _snapshot(self, domains: Domain) -> Domain:
        return {k: set(v) for k, v in domains.items()}

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------
    def _backtrack(self, grid: Grid, domains: Domain) -> bool:
        cell = self._select_unassigned(domains, grid)
        if cell is None:
            self._n_solutions += 1
            if self._solution is None:
                self._solution = deepcopy(grid)
            if self.uniqueness_check and self._n_solutions > 1:
                return False
            return not self.uniqueness_check

        r, c = cell
        for val in sorted(domains[(r, c)]):
            # Pure BT consistency check
            if not self.use_fc and not self.use_ac3:
                conflict = any(grid[pr][pc] == val for pr, pc in self._peers(r, c))
                if conflict:
                    continue

            self.assignments += 1
            grid[r][c] = val
            saved_domain = set(domains[(r, c)])
            domains[(r, c)] = {val}

            consistent = True
            removals = None
            domain_snapshot = None

            if self.use_fc and not self.use_ac3:
                removals = self._forward_check(domains, r, c, val)
                if removals is None:
                    consistent = False

            if self.use_ac3:
                # Maintaining Arc Consistency (MAC)
                domain_snapshot = self._snapshot(domains)
                if not self._ac3_after_assignment(domains, (r, c)):
                    consistent = False

            if consistent:
                if self._backtrack(grid, domains):
                    if not self.uniqueness_check:
                        return True
                    if self._n_solutions > 1:
                        # uniqueness failed – undo and continue
                        grid[r][c] = 0
                        if domain_snapshot is not None:
                            domains.clear()
                            domains.update(domain_snapshot)
                        elif removals is not None:
                            self._undo(domains, removals)
                            domains[(r, c)] = saved_domain
                        else:
                            domains[(r, c)] = saved_domain
                        return False

            # undo
            grid[r][c] = 0
            if domain_snapshot is not None:
                domains.clear()
                domains.update(domain_snapshot)
            elif removals is not None:
                self._undo(domains, removals)
                domains[(r, c)] = saved_domain
            else:
                domains[(r, c)] = saved_domain
            self.backtracks += 1

        return self.uniqueness_check and self._n_solutions == 1

    def _has_duplicate_givens(self, grid: Grid) -> bool:
        for r in range(9):
            seen = set()
            for c in range(9):
                v = grid[r][c]
                if v != 0:
                    if v in seen:
                        return True
                    seen.add(v)
        for c in range(9):
            seen = set()
            for r in range(9):
                v = grid[r][c]
                if v != 0:
                    if v in seen:
                        return True
                    seen.add(v)
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                seen = set()
                for i in range(3):
                    for j in range(3):
                        v = grid[br + i][bc + j]
                        if v != 0:
                            if v in seen:
                                return True
                            seen.add(v)
        return False

    def solve(self, grid: Grid) -> Optional[Grid]:
        self.reset_counters()
        work = deepcopy(grid)

        if self._has_duplicate_givens(work):
            return None

        domains = self._init_domains(work)

        # Preprocessing AC-3
        if self.use_ac3:
            if not self.ac3(domains):
                return None
            for (r, c), dom in domains.items():
                if len(dom) == 1 and work[r][c] == 0:
                    work[r][c] = next(iter(dom))

        success = self._backtrack(work, domains)
        if self.uniqueness_check:
            return self._solution if self._n_solutions == 1 else None
        return self._solution if success else None

    def is_solvable(self, grid: Grid) -> bool:
        return self.solve(grid) is not None


def solve_sudoku(
    grid: Grid,
    use_fc: bool = True,
    use_ac3: bool = True,
    uniqueness_check: bool = False,
) -> Tuple[Optional[Grid], dict]:
    solver = SudokuCSP(use_fc=use_fc, use_ac3=use_ac3, uniqueness_check=uniqueness_check)
    sol = solver.solve(grid)
    stats = {
        "assignments": solver.assignments,
        "backtracks": solver.backtracks,
        "solved": sol is not None,
    }
    return sol, stats


if __name__ == "__main__":
    puzzle = [
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
    for name, fc, ac3 in [("BT", False, False), ("BT+FC", True, False), ("BT+FC+AC3", True, True)]:
        sol, stats = solve_sudoku(puzzle, use_fc=fc, use_ac3=ac3)
        print(name, "solved=", sol is not None, stats)