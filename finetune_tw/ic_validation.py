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


def validate_predictor_ic(
    predict_batch_fn,
    actual_lookup,
    val_universe,
    val_dates,
    cfg,
    build_ctx_fn,
    batch_size=64,
) -> float:
    """Mean cross-sectional rank IC over h1..cfg.val_ic_horizons on a val subset."""
    pred_len = cfg.pred_len
    horizons = min(cfg.val_ic_horizons, pred_len)
    per_group = {}

    for date in val_dates:
        syms = []
        dfs = []
        x_timestamps = []
        y_timestamps = []
        last_dates = []
        ctx_closes = []

        for sym in val_universe:
            built = build_ctx_fn(sym, date)
            if built is None:
                continue
            ctx_df, x_ts, y_ts, last_date, ctx_close = built
            syms.append(sym)
            dfs.append(ctx_df)
            x_timestamps.append(x_ts)
            y_timestamps.append(y_ts)
            last_dates.append(last_date)
            ctx_closes.append(ctx_close)

        rows = []
        for start in range(0, len(syms), batch_size):
            stop = start + batch_size
            preds = predict_batch_fn(
                dfs[start:stop],
                x_timestamps[start:stop],
                y_timestamps[start:stop],
                pred_len,
            )
            for offset, pred in enumerate(preds):
                if pred is None or len(pred) < pred_len:
                    continue
                index = start + offset
                rows.append(
                    (
                        syms[index],
                        pred["close"].values.astype(float),
                        float(ctx_closes[index]),
                        last_dates[index],
                    )
                )

        for horizon in range(horizons):
            pred_returns = []
            actual_returns = []
            for sym, pred_close, ctx_close, last_date in rows:
                actual_close = np.asarray(
                    actual_lookup(sym, last_date, pred_len),
                    dtype=float,
                )
                if len(actual_close) <= horizon:
                    continue
                pred_returns.append(pred_close[horizon] / ctx_close - 1.0)
                actual_returns.append(actual_close[horizon] / ctx_close - 1.0)

            if len(pred_returns) >= 3:
                per_group[(pd.Timestamp(date), horizon + 1)] = (
                    pred_returns,
                    actual_returns,
                )

    return mean_cross_sectional_ic(per_group)
