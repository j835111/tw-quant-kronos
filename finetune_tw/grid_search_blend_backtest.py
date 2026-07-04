import argparse
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

from finetune_tw.config import Config
from finetune_tw.db import list_symbols
from finetune_tw.backtest import compute_metrics, signals_to_holdings
from finetune_tw.backtest_next_open import (
    _load_price_frames, _load_trading_calendar, _build_signal_and_execution_dates,
    build_next_open_portfolio_returns
)
from finetune_tw.backtest_xgb_embedding import xgb_signals_to_raw_preds

def zscore(s):
    std = s.std()
    if std == 0 or np.isnan(std) or np.isinf(std):
        return s - s.mean()
    return (s - s.mean()) / std

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline grid search for blending weights of Full and Raw models.")
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument(
        "--full_parquet",
        default="finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/diagnostics_full/round6_batch3c_full_scores.parquet"
    )
    parser.add_argument(
        "--raw_parquet",
        default="finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/diagnostics_raw/round6_batch3c_raw_scores.parquet"
    )
    parser.add_argument("--hold_days", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--test_end", default="2026-06-17")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    
    if not Path(args.full_parquet).exists() or not Path(args.raw_parquet).exists():
        print(f"Error: Scored parquet files not found at:\n  {args.full_parquet}\n  {args.raw_parquet}")
        sys.exit(1)
        
    print("Loading scored dataframes...")
    df_full = pd.read_parquet(args.full_parquet, columns=['date', 'symbol', 'score_best'])
    df_raw = pd.read_parquet(args.raw_parquet, columns=['date', 'symbol', 'score_best'])
    
    # Align predictions
    print("Aligning full and raw predictions...")
    df = pd.merge(
        df_full.rename(columns={'score_best': 'score_full'}),
        df_raw.rename(columns={'score_best': 'score_raw'}),
        on=['date', 'symbol'],
        how='inner'
    )
    
    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    trading_dates = _load_trading_calendar(cfg, args.test_end)
    
    variant_signal_dates, variant_execution_dates = _build_signal_and_execution_dates(trading_dates, hold_days=args.hold_days)
    
    print("Loading price frames...")
    price_frames = _load_price_frames(cfg, symbols, args.test_end)
    
    # ========================================================
    # Step 1: Complementarity Diagnostics
    # ========================================================
    print("\n=================== Complementarity Diagnostics ===================")
    
    # 1. Daily Rank Correlation
    rank_corrs = []
    for date, group in df.groupby('date'):
        if len(group) >= 2:
            rank_corrs.append(group['score_full'].corr(group['score_raw'], method='spearman'))
    mean_rank_corr = np.mean(rank_corrs)
    print(f"1. Mean daily cross-sectional Spearman Rank Correlation: {mean_rank_corr:.4f}")
    
    # 2. Daily Top-k Overlap Rate
    overlaps = []
    for date, group in df.groupby('date'):
        if len(group) >= args.top_k:
            top_full = set(group.nlargest(args.top_k, 'score_full')['symbol'])
            top_raw = set(group.nlargest(args.top_k, 'score_raw')['symbol'])
            overlaps.append(len(top_full & top_raw) / args.top_k)
    mean_overlap_rate = np.mean(overlaps)
    print(f"2. Mean daily Top-{args.top_k} overlap rate: {mean_overlap_rate:.2%}")
    
    # Apply daily z-score standardization
    print("\nApplying daily cross-sectional z-score standardization...")
    df['z_full'] = df.groupby('date')['score_full'].transform(zscore)
    df['z_raw'] = df.groupby('date')['score_raw'].transform(zscore)
    
    # ========================================================
    # Step 2: Grid Search Standardized Weights
    # ========================================================
    print("\n=================== Standardized Grid Search (w * Z_full + (1-w) * Z_raw) ===================")
    
    results = []
    daily_returns_dict = {}
    
    weights = np.linspace(0.0, 1.0, 11)
    
    for w in weights:
        # Calculate blended scores using standardized z-scores
        df['score_blended'] = w * df['z_full'] + (1.0 - w) * df['z_raw']
        
        # Build xgb_preds_by_date dict
        xgb_preds_by_date = {}
        for date, group in df.groupby('date'):
            xgb_preds_by_date[date] = dict(zip(group['symbol'], group['score_blended']))
            
        # Run next-open portfolio returns simulation
        raw_preds = xgb_signals_to_raw_preds(xgb_preds_by_date, args.hold_days)
        holdings = signals_to_holdings(raw_preds, variant_signal_dates, args.hold_days, args.top_k, cfg.min_signal_threshold)
        _, daily_returns = build_next_open_portfolio_returns(
            price_frames=price_frames, holdings_sequence=holdings,
            execution_dates=variant_execution_dates, trading_dates=trading_dates
        )
        
        metrics = compute_metrics(daily_returns)
        results.append((w, metrics))
        daily_returns_dict[w] = daily_returns
        
        print(f"Weight w={w:.1f} (full) | Ann={metrics['annualised_return']:7.2%} | Sharpe={metrics['sharpe']:.4f} | MaxDD={metrics['max_drawdown']:7.2%}")
        
    # 3. Strategy Daily Returns Correlation
    ret_corr = daily_returns_dict[1.0].corr(daily_returns_dict[0.0])
    print(f"\n3. Pearson Correlation of daily returns between full (w=1.0) and raw (w=0.0): {ret_corr:.4f}")
    
    print("\n=================== Summary Comparison (Z-Score Standardized) ===================")
    for w, m in results:
        print(f"w={w:.1f} | Sharpe={m['sharpe']:.4f} | MaxDD={m['max_drawdown']:6.2%} | Ann={m['annualised_return']:6.2%}")

if __name__ == "__main__":
    main()
