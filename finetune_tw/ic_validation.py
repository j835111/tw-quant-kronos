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


def _collect_rows_for_date(predict_batch_fn, val_universe, date, cfg, build_ctx_fn, batch_size=64):
    """Return list of (sym, pred_open_arr, pred_open_t1, last_date) for one val date."""
    contexts = []
    for sym in val_universe:
        built = build_ctx_fn(sym, date)
        if built is None:
            continue
        ctx_df, x_ts, y_ts, last_date, _ctx_ref = built
        contexts.append((sym, ctx_df, x_ts, y_ts, last_date))
    return collect_validation_rows_by_date(
        predict_batch_fn,
        {pd.Timestamp(date): contexts},
        cfg,
        batch_size=batch_size,
    ).get(pd.Timestamp(date), [])


def collect_validation_rows_by_date(
    predict_batch_fn,
    contexts_by_date,
    cfg,
    batch_size=64,
    prepared_batch_predict_fn=None,
):
    """Collect predictor rows once per date from prepared validation contexts."""
    required_pred_len = min(getattr(cfg, "val_ic_horizons", cfg.pred_len), cfg.pred_len - 1) + 1
    rows_by_date = {}
    for date, contexts in contexts_by_date.items():
        normalized_date = pd.Timestamp(date)
        if not contexts:
            rows_by_date[normalized_date] = []
            continue

        syms, dfs, x_timestamps, y_timestamps, last_dates = [], [], [], [], []
        x_stamps, y_stamps = [], []
        for context in contexts:
            if len(context) not in (5, 7):
                raise ValueError(
                    "Expected validation context tuples with 5 or 7 fields; "
                    f"got {len(context)} fields."
                )
            sym, ctx_df, x_ts, y_ts, last_date = context[:5]
            syms.append(sym)
            dfs.append(ctx_df)
            x_timestamps.append(x_ts)
            y_timestamps.append(y_ts)
            last_dates.append(last_date)
            if len(context) == 7:
                x_stamps.append(context[5])
                y_stamps.append(context[6])
            else:
                x_stamps.append(None)
                y_stamps.append(None)

        rows = []
        for start in range(0, len(syms), batch_size):
            stop = start + batch_size
            df_slice = dfs[start:stop]
            x_timestamp_slice = x_timestamps[start:stop]
            y_timestamp_slice = y_timestamps[start:stop]
            x_stamp_slice = x_stamps[start:stop]
            y_stamp_slice = y_stamps[start:stop]
            if (
                prepared_batch_predict_fn is not None
                and all(x_stamp is not None for x_stamp in x_stamp_slice)
                and all(y_stamp is not None for y_stamp in y_stamp_slice)
            ):
                preds = prepared_batch_predict_fn(
                    df_slice,
                    x_timestamp_slice,
                    y_timestamp_slice,
                    cfg.pred_len,
                    x_stamp_slice,
                    y_stamp_slice,
                )
            else:
                preds = predict_batch_fn(
                    df_slice,
                    x_timestamp_slice,
                    y_timestamp_slice,
                    cfg.pred_len,
                )
            for offset, pred in enumerate(preds):
                if pred is None:
                    continue
                pred_open = pred["open"].values.astype(float)
                if len(pred_open) < required_pred_len:
                    continue
                i = start + offset
                rows.append((syms[i], pred_open, float(pred["open"].iloc[0]), last_dates[i]))
        rows_by_date[normalized_date] = rows
    return rows_by_date


def compute_validation_metrics_from_rows(
    rows_by_date,
    actual_lookup,
    val_dates,
    cfg,
    target_horizon: int = 5,
    compute_ic: bool = True,
    compute_ic_ir: bool = True,
):
    """Compute mean cross-sectional IC and target-horizon IC-IR from shared rows."""
    normalized_dates = [pd.Timestamp(date) for date in val_dates]
    cached_rows_by_date = {}
    for date in normalized_dates:
        rows = rows_by_date.get(date, [])
        cached_rows = []
        for sym, pred_open, pred_open_t1, last_date in rows:
            actual_open = np.asarray(actual_lookup(sym, last_date, cfg.pred_len), dtype=float)
            cached_rows.append((pred_open, pred_open_t1, actual_open))
        cached_rows_by_date[date] = cached_rows

    val_ic = float("nan")
    if compute_ic:
        horizons = min(cfg.val_ic_horizons, cfg.pred_len - 1)
        per_group = {}
        for date in normalized_dates:
            for horizon in range(horizons):
                pred_returns, actual_returns = [], []
                for pred_open, pred_open_t1, actual_open in cached_rows_by_date[date]:
                    if len(actual_open) <= horizon + 1:
                        continue
                    actual_open_t1 = actual_open[0]
                    if pred_open_t1 <= 0 or actual_open_t1 <= 0:
                        continue
                    pred_returns.append(pred_open[horizon + 1] / pred_open_t1 - 1.0)
                    actual_returns.append(actual_open[horizon + 1] / actual_open_t1 - 1.0)
                if len(pred_returns) >= 3:
                    per_group[(date, horizon + 1)] = (pred_returns, actual_returns)
        val_ic = mean_cross_sectional_ic(per_group)

    ic_ir = float("nan")
    max_horizon = min(target_horizon, cfg.pred_len - 1)
    if compute_ic_ir and max_horizon > 0:
        horizon_idx = max_horizon - 1
        per_date_ic: list[float] = []
        for date in normalized_dates:
            pred_returns, actual_returns = [], []
            for pred_open, pred_open_t1, actual_open in cached_rows_by_date[date]:
                if len(actual_open) <= horizon_idx + 1:
                    continue
                actual_open_t1 = actual_open[0]
                if pred_open_t1 <= 0 or actual_open_t1 <= 0:
                    continue
                pred_returns.append(pred_open[horizon_idx + 1] / pred_open_t1 - 1.0)
                actual_returns.append(actual_open[horizon_idx + 1] / actual_open_t1 - 1.0)
            ic = rank_ic(pred_returns, actual_returns)
            if np.isfinite(ic):
                per_date_ic.append(ic)
        if len(per_date_ic) >= 3:
            arr = np.array(per_date_ic)
            ic_ir = float(arr.mean() / (arr.std() + 1e-8))

    return val_ic, ic_ir


def validate_predictor_ic(
    predict_batch_fn,
    actual_lookup,
    val_universe,
    val_dates,
    cfg,
    build_ctx_fn,
    batch_size=64,
) -> float:
    """Mean cross-sectional rank IC over open-to-open returns on a val subset."""
    rows_by_date = {
        pd.Timestamp(date): _collect_rows_for_date(
            predict_batch_fn,
            val_universe,
            date,
            cfg,
            build_ctx_fn,
            batch_size,
        )
        for date in val_dates
    }
    val_ic, _ = compute_validation_metrics_from_rows(
        rows_by_date,
        actual_lookup,
        val_dates,
        cfg,
        compute_ic=True,
        compute_ic_ir=False,
    )
    return val_ic


def validate_predictor_ic_ir(
    predict_batch_fn,
    actual_lookup,
    val_universe,
    val_dates,
    cfg,
    build_ctx_fn,
    batch_size=64,
    target_horizon: int = 5,
) -> float:
    """IC-IR = mean(IC) / std(IC) at target_horizon across val_dates.

    More noise-robust than IC mean: rewards consistent signal over single-day spikes.
    Returns nan if fewer than 3 finite IC values are available.
    """
    if min(target_horizon, cfg.pred_len - 1) <= 0:
        return float("nan")

    rows_by_date = {
        pd.Timestamp(date): _collect_rows_for_date(
            predict_batch_fn,
            val_universe,
            date,
            cfg,
            build_ctx_fn,
            batch_size,
        )
        for date in val_dates
    }
    _, ic_ir = compute_validation_metrics_from_rows(
        rows_by_date,
        actual_lookup,
        val_dates,
        cfg,
        target_horizon=target_horizon,
        compute_ic=False,
        compute_ic_ir=True,
    )
    return ic_ir
