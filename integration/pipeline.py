"""
End-to-end pipeline:

  mosaic image
      ↓
  NN reads digits + probabilities
      ↓
  AC-3 / CSP detects conflict
      ↓
  Search proposes cheapest corrections
      ↓
  CSP solves corrected grid
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from csp.sudoku_csp import SudokuCSP, Grid
from nn.infer import load_model, read_mosaic, make_hard_digit
from search.correction_search import CorrectionSearch


class SudokuPipeline:
    def __init__(
        self,
        model_path: str | Path = "models/digit_cnn.pt",
        max_corrections: int = 3,
        device: str = "cpu",
    ):
        self.device = device
        self.model = load_model(model_path, device=device)
        self.searcher = CorrectionSearch(max_corrections=max_corrections)
        self.csp = SudokuCSP(use_fc=True, use_ac3=True)

    def solve_from_mosaic(
        self,
        mosaic: np.ndarray,
        algorithm: str = "ucs",
    ) -> dict:
        """
        Full pipeline from a 252×252 mosaic.

        Returns a rich result dict.
        """
        # 1. NN reading
        read_grid, probs, confidence = read_mosaic(mosaic, self.model, device=self.device)

        # 2. Quick check with AC-3 / CSP
        solvable = self.csp.is_solvable(read_grid)

        result = {
            "read_grid": read_grid,
            "confidence": confidence,
            "probs": {f"{r},{c}": v for (r, c), v in probs.items()},
            "initially_solvable": solvable,
            "corrections": [],
            "corrected_grid": None,
            "solution": None,
            "search_stats": {},
            "csp_stats": {},
            "status": "ok",
        }

        if solvable:
            sol, stats = self.csp.solve(read_grid), {
                "assignments": self.csp.assignments,
                "backtracks": self.csp.backtracks,
            }
            # re-solve to get stats cleanly
            sol, stats = _solve_with_stats(read_grid)
            result["solution"] = sol
            result["corrected_grid"] = read_grid
            result["csp_stats"] = stats
            return result

        # 3. Search for corrections
        corrected, corrs, search_stats = self.searcher.search(
            read_grid, probs, algorithm=algorithm
        )
        result["search_stats"] = search_stats
        result["corrections"] = [
            {"cell": [r, c], "new_digit": d} for r, c, d in corrs
        ]

        if corrected is None:
            result["status"] = "unreadable"
            # list least confident cells
            flat = [
                (confidence[r][c], r, c)
                for r in range(9) for c in range(9)
                if read_grid[r][c] != 0
            ]
            flat.sort()
            result["least_confident_cells"] = [
                {"cell": [r, c], "confidence": conf} for conf, r, c in flat[:5]
            ]
            return result

        # 4. Solve corrected puzzle
        sol, csp_stats = _solve_with_stats(corrected)
        result["corrected_grid"] = corrected
        result["solution"] = sol
        result["csp_stats"] = csp_stats
        if sol is None:
            result["status"] = "unreadable"
        return result


def _solve_with_stats(grid: Grid):
    solver = SudokuCSP(use_fc=True, use_ac3=True)
    sol = solver.solve(grid)
    stats = {
        "assignments": solver.assignments,
        "backtracks": solver.backtracks,
        "solved": sol is not None,
    }
    return sol, stats


def build_mosaic_from_grid(
    grid: Grid,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    model,
    seed: int = 0,
    inject: Optional[List[Tuple[int, int, int]]] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Build a 252×252 mosaic.
    `inject` is a list of (r, c, wrong_digit) that forces a wrong image.
    """
    rng = np.random.default_rng(seed)
    mosaic = np.zeros((252, 252), dtype=np.float32)

    for r in range(9):
        for c in range(9):
            true_d = grid[r][c]
            if true_d == 0:
                continue
            # decide which digit image to place
            place_d = true_d
            if inject:
                for ir, ic, wd in inject:
                    if ir == r and ic == c:
                        place_d = wd
                        break
            # pick an image of place_d (possibly hard)
            if place_d != true_d:
                patch = make_hard_digit(
                    place_d, X_pool, y_pool, model, device=device,
                    max_confidence=0.65, rng=rng,
                )
            else:
                mask = y_pool == (place_d - 1)
                candidates = X_pool[mask]
                idx = rng.integers(0, len(candidates))
                patch = candidates[idx]
            mosaic[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = patch
    return mosaic