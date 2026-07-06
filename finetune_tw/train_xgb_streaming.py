"""Memory-safe Round-6-followup retraining: TWSE-filtered, streamed XGBoost LambdaRankIC (Batch 2).

Why this exists instead of train_xgb_lambdarank.train():
- embeddings_train.parquet is 11GB of float64; train()'s `df[feat_cols].values` needs ~14GB
  and this box has 13GB RAM. We stream row batches into a QuantileDMatrix instead
  (binned representation, ~2GB), statistically equivalent under tree_method=hist.
- Rows must only be *contiguous per date* for the group-wise objective, not globally
  sorted. All three cached parquets satisfy this in file order (verified: every date is
  exactly one run), so group sizes are computed in stream order and no sort is needed.
- The pairwise objective is O(n^2) per date on one core; groups are independent, and the
  heavy numpy ops release the GIL, so a thread pool parallelizes it without changing the math.

Training protocol (params, rounds, early stopping, eval metric) is identical to Round 6.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from finetune_tw.lambdarank_ic import lambdarank_ic_grad_hess
from finetune_tw.round6_diagnostics import (
    batch_to_matrix, feature_set_columns, _date_strings,
)
from finetune_tw.trading_calendar import twse_trading_days
from finetune_tw.train_xgb_lambdarank import rank_ic_eval_metric

BATCH_SIZE = 100_000
DEFAULT_MEM_LIMIT_GB = 10.0
DEFAULT_MALLOC_ARENA_MAX = "2"
MEM_LIMIT_ENV_VAR = "KRONOS_XGB_MEM_LIMIT_GB"
MALLOC_ARENA_MAX_ENV_VAR = "MALLOC_ARENA_MAX"
MALLOC_ARENA_MAX_DEFAULT_ENV_VAR = "KRONOS_XGB_MALLOC_ARENA_MAX"
BYTES_PER_GB = 1024 ** 3
SELECTION_METRICS = ("rank_ic", "top_k_excess", "ndcg_at_k")


def _parse_positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number, got {value!r}") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return parsed


def _parse_positive_int(value: str, name: str) -> str:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return str(parsed)


def _resolve_mem_limit_gb(cli_value: float | None) -> float:
    if cli_value is not None:
        if not np.isfinite(cli_value) or cli_value <= 0:
            raise ValueError(f"--mem-limit-gb must be a positive number, got {cli_value!r}")
        return cli_value
    env_value = os.environ.get(MEM_LIMIT_ENV_VAR)
    if env_value is None:
        return DEFAULT_MEM_LIMIT_GB
    return _parse_positive_float(env_value, MEM_LIMIT_ENV_VAR)


def _apply_memory_limit(limit_gb: float) -> None:
    if resource is None:
        raise RuntimeError("resource.RLIMIT_AS is unavailable on this platform")
    limit_bytes = int(limit_gb * BYTES_PER_GB)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"failed to set RLIMIT_AS to {limit_gb:g} GB ({limit_bytes} bytes)"
        ) from exc


def _self_exec_argv() -> list[str]:
    if __spec__ is not None and __spec__.name:
        return [sys.executable, "-m", __spec__.name, *sys.argv[1:]]
    return [sys.executable, *sys.argv]


def _ensure_malloc_arena_max_for_cli() -> None:
    if os.environ.get(MALLOC_ARENA_MAX_ENV_VAR) is not None:
        return
    try:
        arena_max = _parse_positive_int(
            os.environ.get(MALLOC_ARENA_MAX_DEFAULT_ENV_VAR, DEFAULT_MALLOC_ARENA_MAX),
            MALLOC_ARENA_MAX_DEFAULT_ENV_VAR,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    env = os.environ.copy()
    env[MALLOC_ARENA_MAX_ENV_VAR] = arena_max
    os.execvpe(sys.executable, _self_exec_argv(), env)


def resolve_date_filter(
    start: str | None,
    end: str | None,
    trading_days: set[str] | None,
) -> set[str] | None:
    """Resolve an optional inclusive date range into a keep-date set.

    If no range is provided, reuse the full trading-day filter when available so the
    existing TWSE-only behavior remains unchanged.
    """
    if start is None and end is None:
        return trading_days
    if start is None or end is None:
        raise ValueError("date filter requires both start and end")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"date filter start must be <= end, got {start!r} > {end!r}")
    if trading_days is not None:
        return {day for day in trading_days if start <= day <= end}
    return {
        day.strftime("%Y-%m-%d")
        for day in pd.date_range(start_ts, end_ts, freq="D")
    }


def _parse_range_arg(value: str, name: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{name} must be formatted START:END, got {value!r}")
    return parts[0], parts[1]


def resolve_multi_date_filter(
    ranges: list[tuple[str, str]],
    trading_days: set[str] | None,
) -> set[str] | None:
    """Union of resolve_date_filter across multiple (start, end) ranges.

    Used to build a non-contiguous validation window (e.g. a momentum-regime
    slice plus a separate later slice) from several disjoint date ranges.
    """
    if not ranges:
        return None
    keep: set[str] = set()
    for start, end in ranges:
        resolved = resolve_date_filter(start, end, trading_days)
        if resolved:
            keep |= resolved
    return keep


def _date_bounds(keep_dates: set[str] | None) -> tuple[str | None, str | None]:
    if not keep_dates:
        return None, None
    return min(keep_dates), max(keep_dates)


def scan_group_sizes(parquet_path, keep_dates: set[str] | None) -> list[int]:
    """Group sizes in stream order, applying the same date filter the iterator will apply.

    Also validates the contiguity precondition: every kept date must be a single run.
    """
    return [size for _, size in _scan_date_runs(parquet_path, keep_dates)]


def _scan_date_runs(parquet_path, keep_dates: set[str] | None) -> list[tuple[str, int]]:
    pf = pq.ParquetFile(parquet_path)
    runs: list[tuple[str, int]] = []
    seen_dates: set[str] = set()
    current_date: str | None = None
    current_size = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["date"]):
        dates = _date_strings(batch.column(0).to_pandas())
        if keep_dates is not None:
            dates = dates[dates.isin(keep_dates)]
        for date in dates.to_numpy():
            if current_date is None:
                current_date = date
                current_size = 1
                continue
            if date == current_date:
                current_size += 1
                continue
            seen_dates.add(current_date)
            if date in seen_dates:
                raise ValueError(f"{parquet_path}: some dates are split into non-contiguous runs")
            runs.append((current_date, current_size))
            current_date = date
            current_size = 1

    if current_date is not None:
        runs.append((current_date, current_size))
    return runs


def _merge_date_runs(parquet_paths, keep_dates: set[str] | None) -> list[tuple[str, int]]:
    merged: list[tuple[str, int]] = []
    seen_dates: set[str] = set()

    for parquet_path in parquet_paths:
        for date, size in _scan_date_runs(parquet_path, keep_dates):
            if merged and merged[-1][0] == date:
                prev_date, prev_size = merged[-1]
                merged[-1] = (prev_date, prev_size + size)
                continue
            if date in seen_dates:
                raise ValueError(f"{parquet_path}: some dates are split into non-contiguous runs")
            if merged:
                seen_dates.add(merged[-1][0])
            merged.append((date, size))
    return merged


class ParquetIter(xgb.DataIter):
    """Stream (features, label) batches from a parquet file for QuantileDMatrix."""

    def __init__(
        self,
        parquet_path,
        feat_cols: list[str],
        keep_dates: set[str] | None,
        batch_size: int = BATCH_SIZE,
        mh_enabled: bool = False,
        w_h1: float = 0.5,
        w_h3: float = 0.3,
        w_h5: float = 0.2,
        targets_path: str | None = None,
    ):
        self._path = parquet_path
        self._feat_cols = feat_cols
        self._keep_dates = keep_dates
        self._batch_size = batch_size
        self._mh_enabled = mh_enabled
        self._w_h1 = w_h1
        self._w_h3 = w_h3
        self._w_h5 = w_h5
        self._targets_path = targets_path
        self._batches = None
        
        self._targets_df = None
        if self._mh_enabled and self._targets_path:
            print(f"Loading Multi-Horizon train targets in-memory from {self._targets_path}...", flush=True)
            self._targets_df = pd.read_parquet(self._targets_path)
            self._targets_df['date'] = _date_strings(self._targets_df['date'])
            self._targets_df = self._targets_df.set_index(['date', 'symbol']).sort_index()
            
        super().__init__()

    def reset(self) -> None:
        pf = pq.ParquetFile(self._path)
        cols = ["date", "symbol", "label", *self._feat_cols] if self._mh_enabled else ["date", "label", *self._feat_cols]
        self._batches = pf.iter_batches(
            batch_size=self._batch_size,
            columns=cols,
        )

    def next(self, input_data) -> bool:
        if self._batches is None:
            self.reset()
        for batch in self._batches:
            name_to_idx = {name: i for i, name in enumerate(batch.schema.names)}
            row_idx = None
            if self._keep_dates is not None:
                dates = _date_strings(batch.column(name_to_idx["date"]).to_pandas())
                mask = dates.isin(self._keep_dates).to_numpy()
                if not mask.any():
                    continue
                row_idx = np.flatnonzero(mask)
            
            x = batch_to_matrix(batch, self._feat_cols, row_idx=row_idx)
            
            if self._mh_enabled and self._targets_df is not None:
                dates = _date_strings(batch.column(name_to_idx["date"]).to_pandas())
                symbols = batch.column(name_to_idx["symbol"]).to_pandas()
                keys = list(zip(dates, symbols))
                try:
                    mh_rows = self._targets_df.loc[keys]
                    y_h1 = mh_rows['label_h1'].values
                    y_h3 = mh_rows['label_h3'].values
                    y_h5 = mh_rows['label_h5'].values
                    y = self._w_h1 * y_h1 + self._w_h3 * y_h3 + self._w_h5 * y_h5
                except KeyError:
                    y = batch.column(name_to_idx["label"]).to_numpy(zero_copy_only=False)
            else:
                y = batch.column(name_to_idx["label"]).to_numpy(zero_copy_only=False)
                
            if row_idx is not None:
                y = y[row_idx]
            input_data(data=x, label=y.astype(np.float32))
            return True
        return False


def load_val_matrix(
    parquet_path,
    feat_cols: list[str],
    keep_dates: set[str] | None,
    mh_enabled: bool = False,
    w_h1: float = 0.5,
    w_h3: float = 0.3,
    w_h5: float = 0.2,
    targets_path: str | None = None,
) -> tuple[xgb.DMatrix, list[int]]:
    """Validation set is small enough (~120k rows) for an in-memory float32 DMatrix."""
    paths = list(parquet_path) if isinstance(parquet_path, (list, tuple)) else [parquet_path]
    if not paths:
        raise ValueError("at least one validation parquet is required")
    groups = [size for _, size in _merge_date_runs(paths, keep_dates)]
    n_rows = sum(groups)
    x = np.empty((n_rows, len(feat_cols)), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32)

    pos = 0
    for path in paths:
        pos = _fill_matrix_from_parquet(
            path,
            feat_cols,
            keep_dates,
            x,
            y,
            pos,
            mh_enabled=mh_enabled,
            w_h1=w_h1,
            w_h3=w_h3,
            w_h5=w_h5,
            targets_path=targets_path,
        )
    assert pos == n_rows
    dval = xgb.DMatrix(x, label=y)
    dval.set_group(groups)
    return dval, groups


def _fill_matrix_from_parquet(
    parquet_path,
    feat_cols: list[str],
    keep_dates: set[str] | None,
    x: np.ndarray,
    y: np.ndarray,
    pos: int,
    mh_enabled: bool = False,
    w_h1: float = 0.5,
    w_h3: float = 0.3,
    w_h5: float = 0.2,
    targets_path: str | None = None,
) -> int:
    df_targets = None
    if mh_enabled and targets_path:
        df_targets = pd.read_parquet(targets_path)
        df_targets['date'] = _date_strings(df_targets['date'])
        df_targets = df_targets.set_index(['date', 'symbol']).sort_index()

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["date", "symbol", "label", *feat_cols],
    ):
        name_to_idx = {name: i for i, name in enumerate(batch.schema.names)}
        row_idx = None
        n_rows = batch.num_rows
        if keep_dates is not None:
            dates = _date_strings(batch.column(name_to_idx["date"]).to_pandas())
            mask = dates.isin(keep_dates).to_numpy()
            if not mask.any():
                continue
            row_idx = np.flatnonzero(mask)
            n_rows = len(row_idx)
            
        x[pos:pos + n_rows] = batch_to_matrix(batch, feat_cols, row_idx=row_idx)
        
        if mh_enabled and df_targets is not None:
            dates = _date_strings(batch.column(name_to_idx["date"]).to_pandas())
            symbols = batch.column(name_to_idx["symbol"]).to_pandas()
            if row_idx is not None:
                dates = dates.iloc[row_idx]
                symbols = symbols.iloc[row_idx]
            
            keys = list(zip(dates, symbols))
            try:
                mh_rows = df_targets.loc[keys]
                y_h1 = mh_rows['label_h1'].values
                y_h3 = mh_rows['label_h3'].values
                y_h5 = mh_rows['label_h5'].values
                labels = w_h1 * y_h1 + w_h3 * y_h3 + w_h5 * y_h5
            except KeyError:
                labels = batch.column(name_to_idx["label"]).to_numpy(zero_copy_only=False)
                if row_idx is not None:
                    labels = labels[row_idx]
            y[pos:pos + n_rows] = labels
        else:
            labels = batch.column(name_to_idx["label"]).to_numpy(zero_copy_only=False)
            y[pos:pos + n_rows] = labels[row_idx] if row_idx is not None else labels
            
        pos += n_rows
    return pos


def parallel_lambdarank_ic_objective(group_sizes: list[int], sigma: float = 1.0,
                                     n_threads: int = 8):
    """Same math as lambdarank_ic.lambdarank_ic_objective, with the independent per-date
    grad/hess computations fanned out over a thread pool (numpy releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    boundaries = np.cumsum([0] + list(group_sizes))
    spans = list(zip(boundaries[:-1], boundaries[1:]))
    pool = ThreadPoolExecutor(max_workers=n_threads)

    def _obj(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label()
        grad = np.zeros_like(preds, dtype=np.float64)
        hess = np.zeros_like(preds, dtype=np.float64)

        def work(span):
            start, end = span
            return start, end, lambdarank_ic_grad_hess(preds[start:end], labels[start:end], sigma=sigma)

        for start, end, (g, h) in pool.map(work, spans):
            grad[start:end] = g
            hess[start:end] = h
        return grad, hess

    def close() -> None:
        pool.shutdown(wait=True)

    _obj.close = close  # type: ignore[attr-defined]
    return _obj


def grouped_top_k_excess(
    preds: np.ndarray,
    labels: np.ndarray,
    group_sizes: list[int],
    top_k: int = 10,
) -> float:
    boundaries = np.cumsum([0] + list(group_sizes))
    values = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group_labels = labels[start:end]
        if len(group_labels) < top_k:
            continue
        order = np.argsort(preds[start:end])[::-1][:top_k]
        values.append(float(np.mean(group_labels[order]) - np.mean(group_labels)))
    return float(np.mean(values)) if values else float("nan")


def grouped_ndcg_at_k(
    preds: np.ndarray,
    labels: np.ndarray,
    group_sizes: list[int],
    top_k: int = 10,
) -> float:
    boundaries = np.cumsum([0] + list(group_sizes))
    scores = []
    discounts = 1.0 / np.log2(np.arange(2, top_k + 2, dtype=np.float64))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group_labels = labels[start:end]
        n = len(group_labels)
        if n < top_k:
            continue
        label_order = np.argsort(group_labels)[::-1]
        gains = np.empty(n, dtype=np.float64)
        gains[label_order] = np.arange(n, 0, -1, dtype=np.float64)

        pred_order = np.argsort(preds[start:end])[::-1][:top_k]
        ideal_order = label_order[:top_k]
        ideal_dcg = float(np.sum(gains[ideal_order] * discounts))
        if ideal_dcg <= 0:
            continue
        dcg = float(np.sum(gains[pred_order] * discounts))
        scores.append(dcg / ideal_dcg)
    return float(np.mean(scores)) if scores else float("nan")


def train_streaming(
    train_path,
    val_path,
    feature_set: str,
    keep_dates: set[str] | None = None,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
    params: dict | None = None,
    n_threads: int = 8,
    selection_metric: str = "rank_ic",
    top_k: int = 10,
    train_keep_dates: set[str] | None = None,
    val_keep_dates: set[str] | None = None,
    mh_enabled: bool = False,
    w_h1: float = 0.5,
    w_h3: float = 0.3,
    w_h5: float = 0.2,
    train_targets_path: str | None = None,
    val_targets_path: str | None = None,
    val_use_composite: bool = False,
) -> tuple[xgb.Booster, dict]:
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {SELECTION_METRICS}, got {selection_metric!r}"
        )
    if keep_dates is not None and (
        train_keep_dates is not None or val_keep_dates is not None
    ):
        raise ValueError(
            "keep_dates cannot be combined with train_keep_dates/val_keep_dates"
        )
    effective_train_keep_dates = train_keep_dates if train_keep_dates is not None else keep_dates
    effective_val_keep_dates = val_keep_dates if val_keep_dates is not None else keep_dates
    all_columns = pq.ParquetFile(train_path).schema_arrow.names
    feat_cols = feature_set_columns(all_columns, feature_set)

    train_groups = scan_group_sizes(train_path, effective_train_keep_dates)
    print(f"[{feature_set}] {len(feat_cols)} features, {sum(train_groups)} train rows, "
          f"{len(train_groups)} train dates", flush=True)

    default_params = {
        "max_depth": 4, "eta": 0.05, "tree_method": "hist",
        "max_bin": 64, "nthread": 4,
    }
    train_params = {**default_params, **(params or {})}
    train_params["max_bin"] = int(train_params["max_bin"])

    dtrain = xgb.QuantileDMatrix(
        ParquetIter(
            train_path,
            feat_cols,
            effective_train_keep_dates,
            mh_enabled=mh_enabled,
            w_h1=w_h1,
            w_h3=w_h3,
            w_h5=w_h5,
            targets_path=train_targets_path,
        ),
        max_bin=train_params["max_bin"],
        nthread=train_params.get("nthread"),
    )
    
    val_targets = val_targets_path if val_use_composite else None
    dval, val_groups = load_val_matrix(
        val_path,
        feat_cols,
        effective_val_keep_dates,
        mh_enabled=(mh_enabled and val_use_composite),
        w_h1=w_h1,
        w_h3=w_h3,
        w_h5=w_h5,
        targets_path=val_targets,
    )
    print(f"[{feature_set}] {dval.num_row()} val rows, {len(val_groups)} val dates", flush=True)

    obj = parallel_lambdarank_ic_objective(train_groups, sigma=1.0, n_threads=n_threads)

    def _validation_metrics(preds: np.ndarray) -> dict[str, float]:
        return {
            "rank_ic": rank_ic_eval_metric(preds, dval, val_groups),
            "top_k_excess": grouped_top_k_excess(preds, dval.get_label(), val_groups, top_k=top_k),
            "ndcg_at_k": grouped_ndcg_at_k(preds, dval.get_label(), val_groups, top_k=top_k),
        }

    def feval(preds, dmat):
        metrics = _validation_metrics(preds)
        named = [
            ("rank_ic_loss", -metrics["rank_ic"]),
            (f"neg_top{top_k}_excess", -metrics["top_k_excess"]),
            (f"neg_ndcg{top_k}", -metrics["ndcg_at_k"]),
        ]
        order = {
            "rank_ic": ["top_k_excess", "ndcg_at_k", "rank_ic"],
            "top_k_excess": ["rank_ic", "ndcg_at_k", "top_k_excess"],
            "ndcg_at_k": ["rank_ic", "top_k_excess", "ndcg_at_k"],
        }[selection_metric]
        lookup = {
            "rank_ic": named[0],
            "top_k_excess": named[1],
            "ndcg_at_k": named[2],
        }
        return [lookup[key] for key in order]

    try:
        booster = xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
            obj=obj,
            evals=[(dval, "val")],
            custom_metric=feval,
            maximize=False,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10,
        )
    finally:
        close = getattr(obj, "close", None)
        if close is not None:
            close()
    best_iteration = int(booster.best_iteration)
    best_preds = booster.predict(dval, iteration_range=(0, best_iteration + 1))
    best_metrics = _validation_metrics(best_preds)
    train_filter_start, train_filter_end = _date_bounds(effective_train_keep_dates)
    val_filter_start, val_filter_end = _date_bounds(effective_val_keep_dates)
    summary = {
        "feature_set": feature_set,
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "train_rows": sum(train_groups),
        "train_dates": len(train_groups),
        "train_filter_start": train_filter_start,
        "train_filter_end": train_filter_end,
        "val_rows": int(dval.num_row()),
        "val_dates": len(val_groups),
        "val_filter_start": val_filter_start,
        "val_filter_end": val_filter_end,
        "best_iteration": best_iteration,
        "selection_metric": selection_metric,
        "top_k": top_k,
        "best_val_rank_ic": float(best_metrics["rank_ic"]),
        "best_val_top_k_excess": float(best_metrics["top_k_excess"]),
        "best_val_ndcg_at_k": float(best_metrics["ndcg_at_k"]),
        "best_val_selection_score": float(best_metrics[selection_metric]),
    }
    return booster, summary


def main() -> None:
    artifact_dir = Path("finetune_tw/outputs/tw_daily/round6_artifacts")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(artifact_dir / "embeddings_train.parquet"))
    parser.add_argument("--val", nargs="+", default=[str(artifact_dir / "embeddings_val.parquet")],
                        help="One or more validation parquet paths (e.g. to draw a "
                             "non-contiguous validation window from both the train and "
                             "val source files)")
    parser.add_argument("--db", default="finetune_tw/data/tw_stocks.db")
    parser.add_argument("--features", choices=["full", "emb", "raw"], required=True)
    parser.add_argument("--out", required=True, help="Output path for the booster (.json)")
    parser.add_argument("--no-twse-filter", action="store_true",
                        help="Keep pd.bdate_range phantom days (Round 6 behaviour) instead of filtering")
    parser.add_argument("--num_boost_round", type=int, default=200)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="rank_ic")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--train-start")
    parser.add_argument("--train-end")
    parser.add_argument("--val-start")
    parser.add_argument("--val-end")
    parser.add_argument(
        "--train-exclude-range", action="append", metavar="START:END",
        help="Exclude an inclusive date range from the train filter (repeatable). "
             "Use to carve a slice out of --train-start/--train-end, e.g. when that "
             "slice is used as a separate --val-range instead.",
    )
    parser.add_argument(
        "--val-range", action="append", metavar="START:END",
        help="Add an inclusive date range to the validation filter (repeatable). "
             "Mutually exclusive with --val-start/--val-end; use this to build a "
             "validation window from several disjoint ranges (e.g. a momentum-regime "
             "slice plus a later slice).",
    )
    parser.add_argument(
        "--mem-limit-gb",
        type=float,
        default=None,
        help=(
            f"Set RLIMIT_AS in GiB (default: {DEFAULT_MEM_LIMIT_GB:g}, "
            f"or ${MEM_LIMIT_ENV_VAR} if set)"
        ),
    )
    parser.add_argument("--mh_enabled", action="store_true", help="Enable Multi-Horizon composite target training")
    parser.add_argument("--w_h1", type=float, default=0.5, help="Weight for 1-day return target")
    parser.add_argument("--w_h3", type=float, default=0.3, help="Weight for 3-day return target")
    parser.add_argument("--w_h5", type=float, default=0.2, help="Weight for 5-day return target")
    parser.add_argument("--train_targets", default=None, help="Path to train multi-horizon targets parquet")
    parser.add_argument("--val_targets", default=None, help="Path to val multi-horizon targets parquet")
    parser.add_argument("--val_use_composite", action="store_true", help="Use composite target for validation metrics as well")
    args = parser.parse_args()
    try:
        _apply_memory_limit(_resolve_mem_limit_gb(args.mem_limit_gb))
    except ValueError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    if args.val_range and (args.val_start or args.val_end):
        parser.error("--val-range cannot be combined with --val-start/--val-end")

    trading_days = None if args.no_twse_filter else twse_trading_days(args.db)
    if trading_days is not None:
        print(f"TWSE filter on: {len(trading_days)} trading days", flush=True)
    try:
        train_keep_dates = resolve_date_filter(args.train_start, args.train_end, trading_days)
        if args.train_exclude_range:
            exclude_ranges = [_parse_range_arg(r, "--train-exclude-range") for r in args.train_exclude_range]
            exclude_dates = resolve_multi_date_filter(exclude_ranges, trading_days)
            if train_keep_dates is None:
                train_keep_dates = trading_days
            if train_keep_dates is not None and exclude_dates:
                train_keep_dates = train_keep_dates - exclude_dates
        if args.val_range:
            val_ranges = [_parse_range_arg(r, "--val-range") for r in args.val_range]
            val_keep_dates = resolve_multi_date_filter(val_ranges, trading_days)
        else:
            val_keep_dates = resolve_date_filter(args.val_start, args.val_end, trading_days)
    except ValueError as exc:
        parser.error(str(exc))

    booster, summary = train_streaming(
        args.train,
        args.val,
        args.features,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        n_threads=args.n_threads,
        selection_metric=args.selection_metric,
        top_k=args.top_k,
        train_keep_dates=train_keep_dates,
        val_keep_dates=val_keep_dates,
        mh_enabled=args.mh_enabled,
        w_h1=args.w_h1,
        w_h3=args.w_h3,
        w_h5=args.w_h5,
        train_targets_path=args.train_targets,
        val_targets_path=args.val_targets,
        val_use_composite=args.val_use_composite,
    )
    booster.save_model(args.out)
    with open(Path(args.out).with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {args.out}  best_iteration={summary['best_iteration']}  "
          f"{summary['selection_metric']}={summary['best_val_selection_score']:.6f}  "
          f"rank-IC={summary['best_val_rank_ic']:.6f}", flush=True)


if __name__ == "__main__":
    _ensure_malloc_arena_max_for_cli()
    main()
