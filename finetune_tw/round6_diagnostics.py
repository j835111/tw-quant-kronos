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
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from finetune_tw.ic_validation import rank_ic
from finetune_tw.train_xgb_lambdarank import _TECH_FEATURE_COLUMNS, _feature_columns

ARTIFACT_DIR = Path("finetune_tw/outputs/tw_daily/round6_artifacts")


def twse_trading_days(db_path, benchmark_symbol: str = "^TWII", min_symbols: int = 500) -> set[str]:
    """Real TWSE trading days = dates with a benchmark row AND >= min_symbols stock rows.

    Either criterion alone is dirty in tw_stocks.db: the benchmark has a spurious row on
    2025-08-01 (only 11 stocks that day), while typhoon closure days (e.g. 2016-09-27/28)
    have provider-emitted stock rows but correctly no benchmark row. The intersection
    drops both failure modes.
    """
    conn = sqlite3.connect(db_path)
    try:
        bench = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM daily_prices WHERE symbol = ?", (benchmark_symbol,)
            )
        }
        busy = {
            r[0] for r in conn.execute(
                "SELECT date FROM daily_prices GROUP BY date HAVING COUNT(*) >= ?", (min_symbols,)
            )
        }
    finally:
        conn.close()
    return bench & busy


def _date_strings(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m-%d")
    return series.astype(str).str.slice(0, 10)


def stream_scores(
    parquet_path,
    booster,
    trading_days: set[str],
    iteration_ranges: dict[str, tuple[int, int]],
    batch_size: int = 40_000,
) -> pd.DataFrame:
    """Filter to real trading days and predict, one record batch at a time.

    Returns a small frame: date, symbol, label, the raw tech features, and one score
    column per entry in iteration_ranges.
    """
    pf = pq.ParquetFile(parquet_path)
    feat_cols = None
    keep = ["date", "symbol", "label", *_TECH_FEATURE_COLUMNS]
    outs = []
    for batch in pf.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        if feat_cols is None:
            feat_cols = _feature_columns(chunk)
        dates = _date_strings(chunk["date"])
        mask = dates.isin(trading_days)
        if not mask.any():
            continue
        sub = chunk.loc[mask]
        x = sub[feat_cols].to_numpy(dtype=np.float32)
        rec = sub[keep].copy()
        rec["date"] = dates.loc[mask]
        for name, it_range in iteration_ranges.items():
            rec[name] = booster.inplace_predict(x, iteration_range=it_range)
        outs.append(rec)
    return pd.concat(outs, ignore_index=True)


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
    parser.add_argument("--best-iteration", type=int, default=190,
                        help="Round 6 best_iteration from the training log (docs/round6_artifact_evaluation.md)")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    days = twse_trading_days(args.db)
    print(f"TWSE calendar: {len(days)} trading days ({min(days)} .. {max(days)})", flush=True)

    booster = xgb.Booster()
    booster.load_model(args.model)
    iteration_ranges = {
        "score_full": (0, 0),  # all trees
        "score_best": (0, args.best_iteration + 1),
    }

    scored = stream_scores(args.parquet, booster, days, iteration_ranges)
    n_dates = scored["date"].nunique()
    print(f"scored {len(scored)} rows on {n_dates} real trading days", flush=True)
    scored.to_parquet(out_dir / "round6_test_scores.parquet", index=False)

    summary: dict = {"n_rows": len(scored), "n_dates": n_dates, "top_k": args.top_k}
    for score_col in iteration_ranges:
        daily = per_day_metrics(scored, score_col=score_col, top_k=args.top_k)
        daily.to_csv(out_dir / f"round6_test_daily_{score_col}.csv", index=False)
        for freq, tag in (("Q", "quarterly"), ("M", "monthly")):
            agg = aggregate_period(daily, freq=freq)
            agg.to_csv(out_dir / f"round6_test_{tag}_{score_col}.csv", index=False)
        overall = aggregate_period(daily.assign(date="2099-01-01"), freq="Y").iloc[0]
        summary[score_col] = {
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
    for feat in _TECH_FEATURE_COLUMNS:
        daily = per_day_metrics(scored, score_col=feat, top_k=args.top_k)
        agg = aggregate_period(daily, freq="Q")
        feat_quarters[feat] = agg
    feat_table = pd.concat(feat_quarters, names=["feature"]).reset_index(level=0)
    feat_table.to_csv(out_dir / "round6_test_feature_quarterly.csv", index=False)

    with open(out_dir / "round6_test_diagnostics.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"outputs written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
