"""
Inference helpers: given a 9×9 mosaic (or list of 28×28 patches),
return the most likely digit grid and per-cell probability distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from nn.model import DigitCNN, set_temperature

Cell = Tuple[int, int]
ProbMap = Dict[Cell, Dict[int, float]]


def load_model(model_path: str | Path, device: str = "cpu") -> DigitCNN:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = DigitCNN()
    model.load_state_dict(ckpt["state_dict"])
    set_temperature(model, ckpt.get("temperature", 1.0))
    model.to(device)
    model.eval()
    return model


def is_blank(patch: np.ndarray, threshold: float = 0.05) -> bool:
    """Simple brightness rule: mean pixel intensity below threshold → blank."""
    return float(patch.mean()) < threshold


def read_mosaic(
    mosaic: np.ndarray,
    model: DigitCNN,
    device: str = "cpu",
    blank_threshold: float = 0.05,
) -> Tuple[List[List[int]], ProbMap, List[List[float]]]:
    """
    mosaic: (252, 252) or (9*28, 9*28) float array in [0,1]

    Returns:
      grid          – 9×9 list of predicted digits (0 = blank)
      probs         – dict (r,c) -> {digit: probability} for non-blank cells
                      digits are 1-9
      confidence    – 9×9 matrix of top-1 confidence (0 for blanks)
    """
    assert mosaic.shape[0] == 252 and mosaic.shape[1] == 252
    grid = [[0] * 9 for _ in range(9)]
    probs: ProbMap = {}
    confidence = [[0.0] * 9 for _ in range(9)]

    patches = []
    coords = []
    for r in range(9):
        for c in range(9):
            patch = mosaic[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28]
            if is_blank(patch, blank_threshold):
                continue
            patches.append(patch)
            coords.append((r, c))

    if not patches:
        return grid, probs, confidence

    X = np.stack(patches)[:, np.newaxis, :, :].astype(np.float32)
    xb = torch.from_numpy(X).to(device)
    with torch.no_grad():
        p = model.predict_proba(xb).cpu().numpy()  # (N, 9)  indices 0..8 → digits 1..9

    for i, (r, c) in enumerate(coords):
        dist = {d + 1: float(p[i, d]) for d in range(9)}
        probs[(r, c)] = dist
        pred = int(p[i].argmax()) + 1
        grid[r][c] = pred
        confidence[r][c] = float(p[i].max())

    return grid, probs, confidence


def make_hard_digit(
    true_digit: int,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    model: DigitCNN,
    device: str = "cpu",
    max_confidence: float = 0.6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Pick a real MNIST image of `true_digit` that the model is uncertain about
    (top-1 confidence ≤ max_confidence) or, if none, the least confident one.
    Used to inject realistic misreads.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    # y_pool is 0-8
    mask = y_pool == (true_digit - 1)
    candidates = X_pool[mask]
    if len(candidates) == 0:
        raise ValueError(f"No images for digit {true_digit}")

    xb = torch.from_numpy(candidates[:, np.newaxis]).float().to(device)
    with torch.no_grad():
        conf = model.predict_proba(xb).max(1).values.cpu().numpy()

    hard_mask = conf <= max_confidence
    if hard_mask.any():
        pool = candidates[hard_mask]
        idx = rng.integers(0, len(pool))
        return pool[idx]
    # fallback: least confident
    idx = int(conf.argmin())
    return candidates[idx]