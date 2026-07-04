from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from finetune_tw.backtest import (
    build_model_specs, compute_metrics, load_predictor_from_spec,
    rank_stocks, signals_to_holdings,
)
from finetune_tw.backtest_data import build_rebalance_inputs, load_symbol_history_frames
from finetune_tw.backtest_next_open import (
    _build_signal_and_execution_dates, _load_price_frames, _load_trading_calendar,
    build_next_open_portfolio_returns,
)
from finetune_tw.config import Config
from finetune_tw.db import list_symbols
from finetune_tw.extract_embeddings import extract_embeddings_batch
from finetune_tw.feature_engineering import add_cross_sectional_rank_features, compute_technical_features
from finetune_tw.backtest_xgb_embedding import _load_model_feature_columns, _assemble_feature_matrix, xgb_signals_to_raw_preds

BATCH_SIZE = 64

def zscore(s):
    std = s.std()
    if std == 0 or np.isnan(std) or np.isinf(std):
        return s - s.mean()
    return (s - s.mean()) / std

def compute_xgb_ensemble_signals(
    predictor,
    booster_full: xgb.Booster,
    booster_raw: xgb.Booster,
    cfg: Config,
    rebal_dates: pd.DatetimeIndex,
    symbols: list[str],
    feat_cols_full: list[str],
    feat_cols_raw: list[str],
    weight: float,
) -> dict[str, dict[str, float]]:
    preload_start = (rebal_dates.min() - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
    preload_end = rebal_dates.max().strftime("%Y-%m-%d")
    history_frames = load_symbol_history_frames(cfg.db_path, symbols, start=preload_start, end=preload_end)

    signals: dict[str, dict[str, float]] = {}
    for i, rebal_date in enumerate(rebal_dates):
        cutoff = rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)
        recent = {sym: frame.loc[frame.index >= cutoff] for sym, frame in history_frames.items()}
        batch_syms, batch_dfs, batch_xts, _ = build_rebalance_inputs(
            recent, symbols, rebal_date, cfg.lookback_window, pred_len=1,
        )

        date_scores: dict[str, float] = {}
        date_embeddings: list[np.ndarray] = []
        date_rows: list[dict[str, float | str]] = []
        ordered_syms: list[str] = []
        
        with torch.no_grad():
            for b in range(0, len(batch_syms), BATCH_SIZE):
                sub_syms = batch_syms[b:b + BATCH_SIZE]
                sub_dfs = batch_dfs[b:b + BATCH_SIZE]
                embeddings = extract_embeddings_batch(predictor, sub_dfs, batch_xts[b:b + BATCH_SIZE])
                tech_rows = []
                for sym, ctx_df in zip(sub_syms, sub_dfs):
                    row = {"date": rebal_date.strftime("%Y-%m-%d"), "symbol": sym}
                    row.update(compute_technical_features(ctx_df))
                    tech_rows.append(row)

                date_embeddings.append(embeddings)
                ordered_syms.extend(sub_syms)
                date_rows.extend(tech_rows)

        if ordered_syms:
            tech_df = add_cross_sectional_rank_features(pd.DataFrame(date_rows))
            embeddings_all = np.vstack(date_embeddings).astype(np.float32, copy=False)
            
            # Predict Full
            features_full = _assemble_feature_matrix(feat_cols_full, embeddings_all, tech_df)
            preds_full = booster_full.predict(xgb.DMatrix(features_full))
            
            # Predict Raw
            features_raw = _assemble_feature_matrix(feat_cols_raw, embeddings_all, tech_df)
            preds_raw = booster_raw.predict(xgb.DMatrix(features_raw))
            
            # Apply daily z-score standardization
            z_full = zscore(preds_full)
            z_raw = zscore(preds_raw)
            
            # Linear Blend
            preds_blended = weight * z_full + (1.0 - weight) * z_raw
            
            for sym, pred in zip(ordered_syms, preds_blended):
                date_scores[sym] = float(pred)

        signals[rebal_date.strftime("%Y-%m-%d")] = date_scores
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i + 1}/{len(rebal_dates)}] {rebal_date.date()}: {len(date_scores)} signals")
            sys.stdout.flush()

    return signals


def run_backtest_xgb_ensemble(
    cfg: Config,
    model_key: str,
    xgb_model_full_path: str,
    xgb_model_raw_path: str,
    weight: float,
    hold_days_list: list[int],
    top_k: int
) -> Path:
    feat_cols_full = _load_model_feature_columns(xgb_model_full_path)
    feat_cols_raw = _load_model_feature_columns(xgb_model_raw_path)
    
    if feat_cols_full is None or feat_cols_raw is None:
        raise ValueError(
            "Ensemble inference requires summary sidecar files (.summary.json) "
            "for both full and raw models containing feature_columns metadata."
        )

    specs = build_model_specs(cfg)
    predictor = load_predictor_from_spec(specs[model_key], cfg)
    
    booster_full = xgb.Booster()
    booster_full.load_model(xgb_model_full_path)
    
    booster_raw = xgb.Booster()
    booster_raw.load_model(xgb_model_raw_path)

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    test_end = str(pd.Timestamp.today().date())
    trading_dates = _load_trading_calendar(cfg, test_end)

    variant_schedules = {hd: _build_signal_and_execution_dates(trading_dates, hold_days=hd) for hd in hold_days_list}
    all_signal_dates = sorted({d for dates, _ in variant_schedules.values() for d in dates})
    signal_dates = pd.DatetimeIndex(all_signal_dates)

    price_frames = _load_price_frames(cfg, symbols, test_end)
    
    print(f"Computing ensemble signals with weight w={weight:.2f} (full) ...")
    xgb_preds_by_date = compute_xgb_ensemble_signals(
        predictor=predictor,
        booster_full=booster_full,
        booster_raw=booster_raw,
        cfg=cfg,
        rebal_dates=signal_dates,
        symbols=symbols,
        feat_cols_full=feat_cols_full,
        feat_cols_raw=feat_cols_raw,
        weight=weight,
    )
    
    del predictor
    torch.cuda.empty_cache()

    out_dir = Path(cfg.output_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    hold_variants: dict[str, dict] = {}
    
    for hd in hold_days_list:
        variant_signal_dates, variant_execution_dates = variant_schedules[hd]
        raw_preds = xgb_signals_to_raw_preds(xgb_preds_by_date, hd)
        holdings = signals_to_holdings(raw_preds, variant_signal_dates, hd, top_k, cfg.min_signal_threshold)
        _, daily_returns = build_next_open_portfolio_returns(
            price_frames=price_frames, holdings_sequence=holdings,
            execution_dates=variant_execution_dates, trading_dates=trading_dates,
        )
        metrics = compute_metrics(daily_returns)
        hold_variants[str(hd)] = {
            "dates": [d.strftime("%Y-%m-%d") for d in daily_returns.index],
            "daily_returns": daily_returns.tolist(),
            "metrics": metrics,
        }
        print(f"  top_k={top_k} hold={hd}d — Ann:{metrics['annualised_return']:.2%} "
              f"Sharpe:{metrics['sharpe']:.2f} DD:{metrics['max_drawdown']:.2%}")

    out_path = out_dir / "backtest_returns_xgb_ensemble_next_open.json"
    out_path.write_text(json.dumps({
        "model_key": model_key,
        "xgb_model_full": xgb_model_full_path,
        "xgb_model_raw": xgb_model_raw_path,
        "weight_full": weight,
        "top_k": top_k,
        "hold_variants": hold_variants
    }, indent=2))
    print(f"\nSaved -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--model", default="pretrained")
    parser.add_argument("--xgb_model_full", required=True)
    parser.add_argument("--xgb_model_raw", required=True)
    parser.add_argument("--weight", type=float, default=0.6, help="Weight for full model (w * z_full + (1-w) * z_raw)")
    parser.add_argument("--hold_days_list", type=int, nargs="+", default=[5])
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    run_backtest_xgb_ensemble(
        cfg=cfg,
        model_key=args.model,
        xgb_model_full_path=args.xgb_model_full,
        xgb_model_raw_path=args.xgb_model_raw,
        weight=args.weight,
        hold_days_list=args.hold_days_list,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
