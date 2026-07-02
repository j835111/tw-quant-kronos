"""
Grid-search top_k × hold_days on a cached inference pass.

Usage:
    # Step 1 — run inference once and cache raw predictions:
    python -m finetune_tw.grid_search_backtest --config ... --model round0 --cache-only

    # Step 2 — grid-search on cached predictions (fast, CPU-only):
    python -m finetune_tw.grid_search_backtest --config ... --model round0 \
        --top_k_list 10 20 30 50 --hold_days_list 3 5 7 10

    # One-shot (inference + grid):
    python -m finetune_tw.grid_search_backtest --config ... --model round0 \
        --top_k_list 10 20 30 50 --hold_days_list 3 5 7 10
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finetune_tw.backtest import (  # noqa: E402
    build_model_specs,
    compute_metrics,
    compute_raw_signals,
    load_predictor_from_spec,
    signals_to_holdings,
    build_portfolio_returns,
    plot_backtest_results,
)
from finetune_tw.config import Config
from finetune_tw.db import query_symbol, list_symbols


def run_grid_search(
    cfg: Config,
    model_key: str,
    top_k_list: list[int],
    hold_days_list: list[int],
    threshold_list: list[float] | None = None,
    cache_only: bool = False,
    load_cache: bool = True,
    rank_h_list: list[int] | None = None,
    max_symbols: int | None = None,
) -> None:
    specs = build_model_specs(cfg)
    if model_key not in specs:
        raise ValueError(f"Unknown model '{model_key}'. Choose: {list(specs)}")
    spec = specs[model_key]

    out_dir = Path(cfg.output_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"raw_preds_{model_key}.json"

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    test_end = str(pd.Timestamp.today().date())
    max_hold = max(hold_days_list)
    min_hold = min(hold_days_list)

    # ── Step 1: inference (or load cache) ────────────────────────────────────
    if load_cache and cache_path.exists():
        print(f"Loading cached raw_preds from {cache_path}")
        cache = json.loads(cache_path.read_text())
        # Reconstruct: {date_str: {sym: pd.Series}}
        raw_preds = {
            date_str: {sym: pd.Series(vals) for sym, vals in syms.items()}
            for date_str, syms in cache["raw_preds"].items()
        }
        fine_dates = pd.DatetimeIndex(cache["fine_dates"])
        print(f"  {len(fine_dates)} periods × {len(symbols)} symbols (cached)")
    else:
        print(f"\nRunning inference: {spec.label}")
        predictor = load_predictor_from_spec(spec, cfg)
        fine_dates = pd.bdate_range(cfg.test_start_date, test_end)[::min_hold]
        print(f"Inference: {len(fine_dates)} periods × {len(symbols)} symbols")
        sys.stdout.flush()
        raw_preds = compute_raw_signals(predictor, cfg, fine_dates, max_hold, symbols)
        del predictor
        torch.cuda.empty_cache()

        # Serialize: pd.Series → list[float]
        serializable = {
            date_str: {sym: s.tolist() for sym, s in syms.items()}
            for date_str, syms in raw_preds.items()
        }
        cache_path.write_text(json.dumps({
            "raw_preds": serializable,
            "fine_dates": [d.strftime("%Y-%m-%d") for d in fine_dates],
        }))
        print(f"  Cached → {cache_path}")

    if cache_only:
        print("Cache written. Exiting (--cache-only).")
        return

    # ── Step 2: load close prices + benchmark ────────────────────────────────
    close_prices: dict[str, pd.Series] = {}
    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=cfg.test_start_date, end=test_end)
        if len(df) > 0:
            close_prices[sym] = pd.Series(df["close"].values,
                                           index=pd.DatetimeIndex(df["date"]))

    bm_df = query_symbol(cfg.db_path, cfg.benchmark_symbol,
                          start=cfg.test_start_date, end=test_end)
    bm_daily = pd.Series(bm_df["close"].values,
                          index=pd.DatetimeIndex(bm_df["date"])).pct_change().dropna()
    bm_metrics = compute_metrics(bm_daily)

    # ── Step 3: grid search ───────────────────────────────────────────────────
    thresholds = threshold_list if threshold_list is not None else [0.0]
    multi_thresh = len(thresholds) > 1
    rh_sweep = rank_h_list if rank_h_list else [None]
    print(f"\n{'='*70}")
    print(f"Grid search: top_k={top_k_list} × hold_days={hold_days_list}"
          + (f" × rank_h={rank_h_list}" if rank_h_list else "")
          + (f" × threshold={thresholds}" if multi_thresh else ""))
    print(f"Benchmark: Sharpe={bm_metrics['sharpe']:.2f}  "
          f"Ann={bm_metrics['annualised_return']:.1%}  "
          f"DD={bm_metrics['max_drawdown']:.1%}")
    print(f"{'='*70}")
    header = (f"{'rank_h':>6} {'top_k':>6} {'hold':>5} {'thr':>6} {'Ann':>8} {'Sharpe':>8} {'MaxDD':>8} {'rank':>5}"
              if multi_thresh else
              f"{'rank_h':>6} {'top_k':>6} {'hold':>5} {'Ann':>8} {'Sharpe':>8} {'MaxDD':>8} {'rank':>5}")
    print(header)
    print("-" * len(header))

    results = []
    for rank_h in rh_sweep:
        for threshold in thresholds:
            for top_k in top_k_list:
                for hd in hold_days_list:
                    step = max(1, hd // min_hold)
                    variant_dates = fine_dates[::step]
                    holdings = signals_to_holdings(raw_preds, variant_dates, hd, top_k, threshold, rank_h=rank_h)
                    _, dr = build_portfolio_returns(close_prices, holdings, variant_dates)
                    m = compute_metrics(dr)
                    results.append({
                        "top_k": top_k,
                        "hold_days": hd,
                        "rank_h": rank_h if rank_h is not None else hd,
                        "threshold": threshold,
                        **m,
                        "daily_returns": dr.tolist(),
                        "dates": [d.strftime("%Y-%m-%d") for d in dr.index],
                    })

    # Sort by Sharpe descending
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    for rank, r in enumerate(results, 1):
        if multi_thresh:
            print(f"{r['rank_h']:>6} {r['top_k']:>6} {r['hold_days']:>5} {r['threshold']:>6.3f} "
                  f"{r['annualised_return']:>7.1%} "
                  f"{r['sharpe']:>8.2f} "
                  f"{r['max_drawdown']:>7.1%} "
                  f"{rank:>5}")
        else:
            print(f"{r['rank_h']:>6} {r['top_k']:>6} {r['hold_days']:>5} "
                  f"{r['annualised_return']:>7.1%} "
                  f"{r['sharpe']:>8.2f} "
                  f"{r['max_drawdown']:>7.1%} "
                  f"{rank:>5}")
    sys.stdout.flush()

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        "model_key": model_key,
        "test_start": cfg.test_start_date,
        "test_end": test_end,
        "top_k_list": top_k_list,
        "hold_days_list": hold_days_list,
        "threshold_list": thresholds,
        "benchmark": {
            "dates": [d.strftime("%Y-%m-%d") for d in bm_daily.index],
            "daily_returns": bm_daily.tolist(),
            "metrics": bm_metrics,
        },
        "results": results,
    }
    json_path = out_dir / f"grid_search_{model_key}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {json_path}")

    # ── Plot best combo ───────────────────────────────────────────────────────
    best = results[0]
    thr_str = f"  thr={best['threshold']:.3f}" if multi_thresh else ""
    print(f"\nBest: top_k={best['top_k']} hold={best['hold_days']}d{thr_str}  "
          f"Sharpe={best['sharpe']:.2f}  Ann={best['annualised_return']:.1%}  "
          f"DD={best['max_drawdown']:.1%}")

    # Build backtest-compatible dict for the top-3 combos
    top3_keys = [f"k{r['top_k']}_h{r['hold_days']}_t{r['threshold']:.3f}" for r in results[:3]]
    plot_data = {
        "model_key": f"grid_{model_key}",
        "model_label": f"{spec.label} grid-search (top-3)",
        "test_start": cfg.test_start_date,
        "test_end": test_end,
        "top_k": best["top_k"],
        "hold_variants": {
            f"k{r['top_k']}_h{r['hold_days']}_t{r['threshold']:.3f}": {
                "dates": r["dates"],
                "daily_returns": r["daily_returns"],
                "metrics": {
                    "annualised_return": r["annualised_return"],
                    "sharpe": r["sharpe"],
                    "max_drawdown": r["max_drawdown"],
                },
            }
            for r in results[:3]
        },
        "benchmark": summary["benchmark"],
    }

    # Override plot_backtest_results to handle string hold keys
    _plot_grid(plot_data, out_dir, model_key)
    return summary


def _plot_grid(data: dict, out_dir: Path, model_key: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    hold_variants = data["hold_variants"]
    bm = data["benchmark"]
    hold_keys = list(hold_variants.keys())
    colors = ["#2196F3", "#FF9800", "#4CAF50"] + ["#9C27B0"] * 10

    bm_dates = pd.DatetimeIndex(bm["dates"])
    bm_cum = (1 + pd.Series(bm["daily_returns"], index=bm_dates)).cumprod()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Grid Search — {data['model_label']}  ({data['test_start']} → {data['test_end']})",
                 fontsize=12, fontweight="bold")

    # Cumulative returns
    ax = axes[0]
    ax.plot(bm_cum.index, bm_cum.values, color="#9E9E9E", lw=1.5, ls="--",
            label=f"^TWII Sharpe={bm['metrics']['sharpe']:.2f}")
    for hk, col in zip(hold_keys, colors):
        v = hold_variants[hk]
        dr = pd.Series(v["daily_returns"], index=pd.DatetimeIndex(v["dates"]))
        cum = (1 + dr).cumprod()
        m = v["metrics"]
        ax.plot(cum.index, cum.values, color=col, lw=1.8,
                label=f"{hk}  Sharpe={m['sharpe']:.2f}  Ann={m['annualised_return']:.1%}")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(1, color="black", lw=0.6, ls=":")
    ax.legend(fontsize=8)
    ax.set_title("Top-3 Combinations — Cumulative Returns")

    # Metrics bars
    ax2 = axes[1]
    x = np.arange(3)
    bar_w = 0.22
    metrics_labels = ["Ann Return", "Sharpe/3", "−Max DD"]
    for i, (hk, col) in enumerate(zip(hold_keys, colors)):
        m = hold_variants[hk]["metrics"]
        vals = [m["annualised_return"], m["sharpe"] / 3, -m["max_drawdown"]]
        ax2.bar(x + (i - 1) * bar_w, vals, bar_w, color=col, label=hk, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(metrics_labels)
    ax2.set_title("Key Metrics")
    ax2.legend(fontsize=8)

    out_path = out_dir / f"grid_search_{model_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--model", required=True,
                        choices=["pretrained", "round0", "round1", "round2", "round3", "round4", "round5"])
    parser.add_argument("--top_k_list",    type=int, nargs="+", default=[10, 20, 30, 50])
    parser.add_argument("--hold_days_list", type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--threshold_list", type=float, nargs="+", default=None,
                        help="Min predicted return thresholds to sweep (default: [0.0])")
    parser.add_argument("--rank_h_list", type=int, nargs="+", default=None,
                        help="Ranking horizons to sweep (default: same as hold_days)")
    parser.add_argument("--max_symbols", type=int, default=None,
                        help="Limit to first N symbols for testing (default: all)")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only run inference and cache; skip grid search")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-run inference even if cache exists")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    run_grid_search(
        cfg=cfg,
        model_key=args.model,
        top_k_list=args.top_k_list,
        hold_days_list=args.hold_days_list,
        threshold_list=args.threshold_list,
        cache_only=args.cache_only,
        load_cache=not args.no_cache,
        rank_h_list=args.rank_h_list,
        max_symbols=args.max_symbols,
    )


if __name__ == "__main__":
    main()
