"""Round 6 Batch-1 diagnostics: TWSE-calendar filtering + test-period per-stock scores + per-period IC.

Implements docs/round6-embedding-xgb-lambdarank-improvements.md 補充評估二 Batch 1:
score the cached test-period embeddings with xgb_round6.json on real TWSE trading days only,
then break rank-IC / top-10 excess down by quarter and month to test the 2026-Q2
reversal-exposure hypothesis.

CPU-only: streams the parquet in batches (see round6_m1_status memory: pd.read_parquet
balloons to ~4x file size; this box has 13GB RAM and embeddings_test.parquet is 3.3GB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finetune_tw.feature_engineering import technical_feature_columns
from finetune_tw.ic_validation import rank_ic
from finetune_tw.trading_calendar import twse_trading_days
from finetune_tw.train_xgb_lambdarank import EMBEDDING_PREFIX

ARTIFACT_DIR = Path("finetune_tw/outputs/tw_daily/round6_artifacts")


def feature_set_columns(all_columns: list[str], feature_set: str) -> list[str]:
    """Column list for a named ablation feature set, in the training order
    (embeddings sorted numerically, then the raw technical features)."""
    emb_cols = sorted([c for c in all_columns if c.startswith(EMBEDDING_PREFIX)],
                      key=lambda c: int(c[len(EMBEDDING_PREFIX):]))
    tech_cols = technical_feature_columns(all_columns)
    if feature_set == "full":
        return emb_cols + tech_cols
    if feature_set == "emb":
        return emb_cols
    if feature_set == "raw":
        return tech_cols
    raise ValueError(f"unknown feature set: {feature_set}")


def _date_strings(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m-%d")
    return series.astype(str).str.slice(0, 10)


def batch_to_matrix(batch, feat_cols: list[str], row_idx: np.ndarray | None = None) -> np.ndarray:
    """Arrow RecordBatch -> float32 feature matrix, column by column.

    ~28x faster than batch.to_pandas()[feat_cols].to_numpy() for 800+ columns (pandas
    block consolidation dominates there). Columns are looked up by name because
    iter_batches returns them in schema order, not request order.
    """
    name_to_idx = {name: i for i, name in enumerate(batch.schema.names)}
    n_rows = len(row_idx) if row_idx is not None else batch.num_rows
    x = np.empty((n_rows, len(feat_cols)), dtype=np.float32)
    for j, name in enumerate(feat_cols):
        col = batch.column(name_to_idx[name]).to_numpy(zero_copy_only=False)
        x[:, j] = col[row_idx] if row_idx is not None else col
    return x


def projected_score_columns(all_columns: list[str]) -> list[str]:
    """Columns that may be decoded into pandas / written to scored parquet."""
    ordered = [
        "date",
        "symbol",
        "label",
        *technical_feature_columns(all_columns),
    ]
    seen = set()
    projected = []
    for col in ordered:
        if col in all_columns and col not in seen:
            projected.append(col)
            seen.add(col)
    return projected


def score_read_columns(all_columns: list[str], feat_cols: list[str]) -> list[str]:
    """Columns needed in the Arrow batch to build features and output rows."""
    ordered = [
        "date",
        "symbol",
        "label",
        *technical_feature_columns(all_columns),
        *feat_cols,
    ]
    seen = set()
    projected = []
    for col in ordered:
        if col in all_columns and col not in seen:
            projected.append(col)
            seen.add(col)
    return projected


def stream_scored_batches(
    parquet_path,
    booster,
    trading_days: set[str],
    iteration_ranges: dict[str, tuple[int, int]],
    batch_size: int = 40_000,
    feat_cols: list[str] | None = None,
):
    """Backward-compatible alias for the streaming public scorer."""
    yield from stream_scores(
        parquet_path,
        booster,
        trading_days,
        iteration_ranges,
        batch_size=batch_size,
        feat_cols=feat_cols,
    )


def stream_scores(
    parquet_path,
    booster,
    trading_days: set[str],
    iteration_ranges: dict[str, tuple[int, int]],
    batch_size: int = 40_000,
    feat_cols: list[str] | None = None,
):
    """Yield filtered, scored batch-sized frames without decoding emb_* into pandas."""
    pf = pq.ParquetFile(parquet_path)
    if feat_cols is None:
        feat_cols = feature_set_columns(pf.schema_arrow.names, "full")
    read_cols = score_read_columns(pf.schema_arrow.names, feat_cols)
    output_cols = projected_score_columns(pf.schema_arrow.names)
    for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
        chunk = batch.select(output_cols).to_pandas()
        dates = _date_strings(chunk["date"])
        mask = dates.isin(trading_days)
        if not mask.any():
            continue
        row_idx = np.flatnonzero(mask.to_numpy())
        x = batch_to_matrix(batch, feat_cols, row_idx=row_idx)
        rec = chunk.loc[mask].copy()
        rec["date"] = dates.loc[mask]
        for name, it_range in iteration_ranges.items():
            rec[name] = booster.inplace_predict(x, iteration_range=it_range)
        yield rec.reset_index(drop=True)


def iter_scored_dates(scored_batches) -> pd.DataFrame:
    """Yield one fully completed scored date at a time when each date is a single contiguous run."""
    pending: pd.DataFrame | None = None
    emitted_dates: set[str] = set()
    for batch in scored_batches:
        if batch.empty:
            continue
        batch_dates = _date_strings(batch["date"]).reset_index(drop=True)
        batch = batch.copy()
        batch["date"] = batch_dates
        combined = batch if pending is None else pd.concat([pending, batch], ignore_index=True)
        dates = combined["date"].to_numpy()
        start = 0
        pending = None
        while start < len(combined):
            date = dates[start]
            end = start + 1
            while end < len(combined) and dates[end] == date:
                end += 1
            date_frame = combined.iloc[start:end].reset_index(drop=True)
            is_last = end == len(combined)
            if date in emitted_dates:
                raise ValueError("scored batch dates must appear in one contiguous run")
            if is_last:
                pending = date_frame
            else:
                emitted_dates.add(date)
                yield date_frame
            start = end
    if pending is not None and not pending.empty:
        date = pending["date"].iloc[0]
        if date in emitted_dates:
            raise ValueError("scored batch dates must appear in one contiguous run")
        yield pending.reset_index(drop=True)


def _concat_frames(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def _write_scored_frame(
    writer: pq.ParquetWriter | None,
    path: Path,
    frame: pd.DataFrame,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer


def _empty_aggregate_row() -> dict[str, float]:
    return {
        "mean_ic": np.nan,
        "ic_ir": np.nan,
        "pct_ic_pos": np.nan,
        "mean_top_excess": np.nan,
        "mean_overlap": np.nan,
    }


def per_day_metrics(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    top_k: int = 10,
) -> pd.DataFrame:
    """Per-date cross-sectional metrics: rank-IC, top-k excess return, and the overlap
    between the model's top-k and the realized top-k (the Codex top-tail metric)."""
    rows = []
    for date, g in df.groupby("date", sort=True):
        if len(g) < top_k * 2:
            continue
        top = g.nlargest(top_k, score_col)
        actual_top = g.nlargest(top_k, label_col)
        universe_mean = g[label_col].mean()
        rows.append({
            "date": date,
            "n": len(g),
            "rank_ic": rank_ic(g[score_col].values, g[label_col].values),
            "top_mean": top[label_col].mean(),
            "universe_mean": universe_mean,
            "top_excess": top[label_col].mean() - universe_mean,
            "overlap_topk": len(set(top["symbol"]) & set(actual_top["symbol"])) / top_k,
        })
    return pd.DataFrame(rows)


def aggregate_period(daily: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    d = daily.copy()
    d["period"] = pd.PeriodIndex(pd.to_datetime(d["date"]), freq=freq).astype(str)
    out = []
    for period, g in d.groupby("period", sort=True):
        ic = g["rank_ic"]
        std = ic.std(ddof=1)
        out.append({
            "period": period,
            "days": len(g),
            "mean_ic": ic.mean(),
            "ic_ir": ic.mean() / std if len(g) > 1 and std > 0 else np.nan,
            "pct_ic_pos": (ic > 0).mean(),
            "mean_top_excess": g["top_excess"].mean(),
            "mean_overlap": g["overlap_topk"].mean(),
            "universe_mean_label": g["universe_mean"].mean(),
        })
    return pd.DataFrame(out)


def main() -> None:
    import xgboost as xgb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=str(ARTIFACT_DIR / "embeddings_test.parquet"))
    parser.add_argument("--model", default=str(ARTIFACT_DIR / "xgb_round6.json"))
    parser.add_argument("--db", default="finetune_tw/data/tw_stocks.db")
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--best-iteration", type=int, default=None,
                        help="Defaults to the model's stored best_iteration; Round 6's is 190 "
                             "(docs/round6_artifact_evaluation.md)")
    parser.add_argument("--features", choices=["full", "emb", "raw"], default="full",
                        help="Feature set the model was trained on (for ablation models)")
    parser.add_argument("--prefix", default="round6_test",
                        help="Output filename prefix, e.g. round6_clean_test for the Batch-2 baseline")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    days = twse_trading_days(args.db)
    print(f"TWSE calendar: {len(days)} trading days ({min(days)} .. {max(days)})", flush=True)

    booster = xgb.Booster()
    booster.load_model(args.model)
    best_iteration = args.best_iteration
    if best_iteration is None:
        best_iteration = getattr(booster, "best_iteration", None)
    iteration_ranges = {"score_full": (0, 0)}  # all trees
    if best_iteration is not None:
        iteration_ranges["score_best"] = (0, int(best_iteration) + 1)

    parquet_columns = pq.ParquetFile(args.parquet).schema_arrow.names
    feat_cols = feature_set_columns(parquet_columns, args.features)
    raw_feat_cols = technical_feature_columns(parquet_columns)
    score_cols = list(iteration_ranges)
    metric_columns = [
        "date",
        "n",
        "rank_ic",
        "top_mean",
        "universe_mean",
        "top_excess",
        "overlap_topk",
    ]
    score_daily_parts = {score_col: [] for score_col in score_cols}
    feat_daily_parts = {feat: [] for feat in raw_feat_cols}
    n_rows = 0
    n_dates = 0
    writer: pq.ParquetWriter | None = None
    scores_path = out_dir / f"{args.prefix}_scores.parquet"

    try:
        scored_batches = stream_scores(
            args.parquet,
            booster,
            days,
            iteration_ranges,
            feat_cols=feat_cols,
        )
        for scored_date in iter_scored_dates(scored_batches):
            n_rows += len(scored_date)
            n_dates += 1
            writer = _write_scored_frame(writer, scores_path, scored_date)
            for score_col in score_cols:
                daily = per_day_metrics(scored_date, score_col=score_col, top_k=args.top_k)
                if not daily.empty:
                    score_daily_parts[score_col].append(daily)
            for feat in feat_daily_parts:
                daily = per_day_metrics(scored_date, score_col=feat, top_k=args.top_k)
                if not daily.empty:
                    feat_daily_parts[feat].append(daily)
    finally:
        if writer is not None:
            writer.close()

    print(f"scored {n_rows} rows on {n_dates} real trading days", flush=True)

    summary: dict = {"n_rows": n_rows, "n_dates": n_dates, "top_k": args.top_k}
    for score_col in score_cols:
        daily = _concat_frames(score_daily_parts[score_col], metric_columns)
        daily.to_csv(out_dir / f"{args.prefix}_daily_{score_col}.csv", index=False)
        for freq, tag in (("Q", "quarterly"), ("M", "monthly")):
            agg = aggregate_period(daily, freq=freq)
            agg.to_csv(out_dir / f"{args.prefix}_{tag}_{score_col}.csv", index=False)
        if daily.empty:
            summary[score_col] = _empty_aggregate_row()
            print(f"[{score_col}] no daily metrics produced", flush=True)
            continue
        overall = aggregate_period(daily.assign(date="2099-01-01"), freq="Y").iloc[0]
        summary[score_col] = _empty_aggregate_row() | {
            "mean_ic": overall["mean_ic"],
            "ic_ir": overall["ic_ir"],
            "pct_ic_pos": overall["pct_ic_pos"],
            "mean_top_excess": overall["mean_top_excess"],
            "mean_overlap": overall["mean_overlap"],
        }
        print(f"[{score_col}] mean IC {overall['mean_ic']:.4f}  IC-IR {overall['ic_ir']:.3f}  "
              f"top-{args.top_k} excess {overall['mean_top_excess'] * 100:+.3f}%  "
              f"overlap {overall['mean_overlap'] * 100:.2f}%", flush=True)

    feat_quarters = {}
    for feat, parts in feat_daily_parts.items():
        daily = _concat_frames(parts, metric_columns)
        feat_quarters[feat] = aggregate_period(daily, freq="Q")
    if feat_quarters:
        feat_table = pd.concat(feat_quarters, names=["feature"]).reset_index(level=0)
    else:
        feat_table = pd.DataFrame(
            columns=["feature", "period", "days", "mean_ic", "ic_ir", "pct_ic_pos",
                     "mean_top_excess", "mean_overlap", "universe_mean_label"]
        )
    feat_table.to_csv(out_dir / f"{args.prefix}_feature_quarterly.csv", index=False)

    with open(out_dir / f"{args.prefix}_diagnostics.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
