"""
python finetune_tw/backtest.py --config finetune_tw/configs/config_tw_daily.yaml
Loads fine-tuned weights from local path or HuggingFace Hub (cfg.hf_repo / cfg.hf_revision).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.config import Config
from finetune_tw.db import query_symbol, list_symbols
from finetune_tw.hf_utils import resolve_src


# ── Pure helper functions (testable without a model) ────────────────────────

def compute_metrics(daily_returns: pd.Series) -> dict:
    ann_ret = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
    cum = (1 + daily_returns).cumprod()
    max_dd = ((cum.cummax() - cum) / cum.cummax()).max()
    return {"annualised_return": float(ann_ret), "sharpe": float(sharpe), "max_drawdown": float(max_dd)}


def rank_stocks(signals: dict[str, float], top_k: int) -> set[str]:
    sorted_syms = sorted(signals, key=signals.__getitem__, reverse=True)
    return set(sorted_syms[:top_k])


def build_portfolio_returns(
    price_data: dict[str, pd.Series],
    holdings_sequence: list[set[str]],
    rebalance_dates: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns:
        period_returns: avg return per hold period, indexed by rebalance_dates[:-1]
        daily_returns:  daily portfolio returns across all hold periods (for accurate metrics)
    """
    period_rets = []
    all_daily: list[pd.Series] = []

    for i in range(len(rebalance_dates) - 1):
        date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1]
        holdings = holdings_sequence[i]

        period_sym_rets = []
        daily_sym_series = []

        for sym in holdings:
            if sym not in price_data:
                continue
            series = price_data[sym]
            mask = (series.index >= date) & (series.index <= next_date)
            sub = series[mask]
            if len(sub) < 2:
                continue
            period_sym_rets.append(sub.iloc[-1] / sub.iloc[0] - 1)
            daily_sym_series.append(sub.pct_change().dropna())

        period_rets.append(float(np.mean(period_sym_rets)) if period_sym_rets else 0.0)

        if daily_sym_series:
            combined = pd.concat(daily_sym_series, axis=1).mean(axis=1)
            all_daily.append(combined)

    period_series = pd.Series(period_rets, index=rebalance_dates[:-1])

    if all_daily:
        daily_returns = pd.concat(all_daily).sort_index()
        daily_returns = daily_returns[~daily_returns.index.duplicated(keep="first")]
    else:
        daily_returns = pd.Series(dtype=float)

    return period_series, daily_returns


# ── Main backtest loop ──────────────────────────────────────────────────────

def run_backtest(cfg: Config) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_dir   = Path(cfg.output_dir) / cfg.exp_name
    tok_path  = exp_dir / "tokenizer" / "best_model"
    pred_path = exp_dir / "predictor" / "best_model"

    tok_src,  tok_kw  = resolve_src(tok_path,  cfg.hf_repo, "tokenizer/best_model",  cfg.hf_revision)
    pred_src, pred_kw = resolve_src(pred_path, cfg.hf_repo, "predictor/best_model", cfg.hf_revision)
    tokenizer = KronosTokenizer.from_pretrained(tok_src,  **tok_kw)
    model     = Kronos.from_pretrained(pred_src, **pred_kw)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
    tokenizer.eval(); model.eval()

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    test_end = str(pd.Timestamp.today().date())

    print(f"Loaded {len(symbols)} symbols, test period {cfg.test_start_date} → {test_end}")
    sys.stdout.flush()

    # Pre-load close prices for all symbols over the test period
    close_prices: dict[str, pd.Series] = {}
    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=cfg.test_start_date, end=test_end)
        if len(df) > 0:
            idx = pd.DatetimeIndex(df["date"])
            close_prices[sym] = pd.Series(df["close"].values, index=idx)

    print(f"Pre-loaded close prices for {len(close_prices)} symbols")
    sys.stdout.flush()

    # Build rebalance dates
    all_dates = pd.bdate_range(cfg.test_start_date, test_end)
    rebalance_dates = all_dates[::cfg.hold_days]
    print(f"Rebalance dates: {len(rebalance_dates)} periods (hold_days={cfg.hold_days})")
    sys.stdout.flush()

    BATCH_SIZE = 64
    holdings_sequence: list[set[str]] = []
    for i, rebal_date in enumerate(rebalance_dates):
        signals: dict[str, float] = {}
        rebal_str = rebal_date.strftime("%Y-%m-%d")
        lookback_start = (rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
        y_ts = pd.date_range(rebal_date, periods=cfg.pred_len, freq="B")

        batch_syms, batch_dfs, batch_xts, batch_yts = [], [], [], []
        for sym in symbols:
            df = query_symbol(cfg.db_path, sym, start=lookback_start, end=rebal_str)
            if len(df) < cfg.lookback_window:
                continue
            ctx = df.iloc[-cfg.lookback_window:]
            ctx_df = ctx[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
            if ctx_df.isnull().any().any():
                continue
            batch_syms.append(sym)
            batch_dfs.append(ctx_df)
            batch_xts.append(pd.to_datetime(ctx["date"]).reset_index(drop=True))
            batch_yts.append(pd.Series(y_ts))

        with torch.no_grad():
            for b in range(0, len(batch_syms), BATCH_SIZE):
                chunk_syms = batch_syms[b:b + BATCH_SIZE]
                chunk_dfs = batch_dfs[b:b + BATCH_SIZE]
                preds = predictor.predict_batch(
                    df_list=chunk_dfs,
                    x_timestamp_list=batch_xts[b:b + BATCH_SIZE],
                    y_timestamp_list=batch_yts[b:b + BATCH_SIZE],
                    pred_len=cfg.pred_len,
                    T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False,
                )
                for sym, pred, ctx_df in zip(chunk_syms, preds, chunk_dfs):
                    if pred is not None and len(pred) >= cfg.pred_len:
                        signals[sym] = pred["close"].iloc[-1] / ctx_df["close"].iloc[-1] - 1

        holdings_sequence.append(rank_stocks(signals, cfg.top_k))
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(rebalance_dates)}] {rebal_str}: {len(signals)} signals, top-{cfg.top_k} selected")
            sys.stdout.flush()

    _, daily_returns = build_portfolio_returns(close_prices, holdings_sequence, rebalance_dates)

    # Benchmark daily returns
    bm_df = query_symbol(cfg.db_path, cfg.benchmark_symbol,
                         start=cfg.test_start_date, end=test_end)
    bm_close = pd.Series(bm_df["close"].values,
                         index=pd.DatetimeIndex(bm_df["date"]))
    bm_daily = bm_close.pct_change().dropna()
    bm_daily = bm_daily.reindex(daily_returns.index).fillna(0)

    metrics    = compute_metrics(daily_returns)
    bm_metrics = compute_metrics(bm_daily)

    print(f"\n=== Backtest Results ({cfg.test_start_date} → {test_end}) ===")
    print(f"Strategy  — Ann. Return: {metrics['annualised_return']:.2%}  "
          f"Sharpe: {metrics['sharpe']:.2f}  Max DD: {metrics['max_drawdown']:.2%}")
    print(f"Benchmark — Ann. Return: {bm_metrics['annualised_return']:.2%}  "
          f"Sharpe: {bm_metrics['sharpe']:.2f}  Max DD: {bm_metrics['max_drawdown']:.2%}")
    sys.stdout.flush()

    # Save metrics JSON
    out_dir = Path(cfg.output_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = {
        "test_start": cfg.test_start_date,
        "test_end": test_end,
        "strategy": metrics,
        "benchmark": bm_metrics,
    }
    (out_dir / "backtest_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    print(f"Metrics saved to {out_dir / 'backtest_metrics.json'}")

    # Plot daily NAV
    cum_strat = (1 + daily_returns).cumprod()
    cum_bm    = (1 + bm_daily).cumprod()
    plt.figure(figsize=(12, 5))
    plt.plot(cum_strat.index, cum_strat.values, label="Kronos-TW Strategy")
    plt.plot(cum_bm.index,    cum_bm.values,    label=cfg.benchmark_symbol, linestyle="--")
    plt.title("Cumulative Return: Strategy vs Benchmark")
    plt.xlabel("Date"); plt.ylabel("Cumulative Return")
    plt.legend(); plt.tight_layout()
    out_path = out_dir / "backtest_result.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--top_k",    type=int,   default=None)
    parser.add_argument("--hold_days", type=int,  default=None)
    parser.add_argument("--pred_len",  type=int,  default=None)
    parser.add_argument("--test_start", default=None)
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.top_k:      cfg.top_k = args.top_k
    if args.hold_days:  cfg.hold_days = args.hold_days
    if args.pred_len:   cfg.pred_len = args.pred_len
    if args.test_start: cfg.test_start_date = args.test_start
    run_backtest(cfg)


if __name__ == "__main__":
    main()
