"""Price-space validation helpers (pure functions + model-driven IC validator)
for selecting predictor checkpoints by forecast skill instead of token CE.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rank_ic(pred, actual) -> float:
    """Spearman rank correlation = Pearson on ranks. No scipy dependency."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 3:
        return float("nan")
    pred_rank = pd.Series(pred[mask]).rank().values
    actual_rank = pd.Series(actual[mask]).rank().values
    if pred_rank.std() < 1e-9 or actual_rank.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(pred_rank, actual_rank)[0, 1])


def mean_cross_sectional_ic(per_group: dict) -> float:
    """per_group: {key: (pred_seq, actual_seq)} -> mean of finite per-group rank_ic."""
    ics = [rank_ic(pred, actual) for (pred, actual) in per_group.values()]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics)) if ics else float("nan")


def pick_val_universe(symbols, n: int, seed: int = 42) -> list:
    """Deterministic subset of symbols for cheap per-epoch validation."""
    syms = sorted(symbols)
    if len(syms) <= n:
        return syms
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(syms), size=n, replace=False)
    return [syms[i] for i in sorted(idx)]


def pick_val_dates(start: str, end: str, n: int) -> list:
    """Evenly spaced business days across [start, end]."""
    bdays = pd.bdate_range(start, end)
    if len(bdays) <= n:
        return list(bdays)
    pos = np.linspace(0, len(bdays) - 1, n).round().astype(int)
    return [bdays[i] for i in sorted(set(pos.tolist()))]


class EarlyStopper:
    """Track the best validation metric and stop after repeated non-improvement."""

    def __init__(self, patience: int = 2, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self._bad = 0

    def update(self, value):
        """Return (is_best, should_stop) for the latest metric value."""
        improved = False
        if value is not None:
            numeric_value = float(value)
            if not math.isnan(numeric_value):
                improved = self.best is None or (
                    numeric_value > self.best
                    if self.mode == "max"
                    else numeric_value < self.best
                )
                if improved:
                    self.best = numeric_value
                    self._bad = 0
                    return True, False

        self._bad += 1
        return False, self._bad > self.patience
