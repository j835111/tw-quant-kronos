"""
signal_today_ensemble.py — Output today's Kronos stock signals using the static Z-Score ensemble blending model.

Usage:
  python -m finetune_tw.signal_today_ensemble \
    --config finetune_tw/configs/config_tw_daily.yaml \
    --xgb_model_full finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/diagnostics_full/round6_batch3c_full.model \
    --xgb_model_raw finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/diagnostics_raw/round6_batch3c_raw.model \
    --weight 0.6
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from finetune_tw.config import Config
from finetune_tw.db import list_symbols, get_last_date, query_symbols_window
from finetune_tw.backtest import (
    build_model_specs,
    load_predictor_from_spec,
    rank_stocks,
)
from finetune_tw.extract_embeddings import extract_embeddings_batch
from finetune_tw.feature_engineering import add_cross_sectional_rank_features, compute_technical_features
from finetune_tw.backtest_xgb_embedding import _load_model_feature_columns, _assemble_feature_matrix

_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
BATCH_SIZE = 64

# Production XGBoost checkpoints are backed up on this HF branch; missing local
# files (fresh clone — finetune_tw/outputs/ is gitignored) are restored from here.
HF_XGB_REPO = "j835111/kronos-tw-finetune"
HF_XGB_REVISION = "round6-batch3c-full-production"
HF_XGB_DIR = "round6_xgb/production"

def _ensure_xgb_model(model_path: str) -> None:
    """Download the XGBoost model and its .summary.json sidecar from HF if missing locally."""
    from huggingface_hub import hf_hub_download

    target = Path(model_path)
    sidecar = target.with_suffix(".summary.json")
    for dst in (target, sidecar):
        if dst.exists():
            continue
        print(f"  {dst.name} not found locally; downloading from {HF_XGB_REPO}@{HF_XGB_REVISION} ...")
        cached = hf_hub_download(
            HF_XGB_REPO,
            f"{HF_XGB_DIR}/{dst.name}",
            revision=HF_XGB_REVISION,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dst)

def zscore(s):
    std = s.std()
    if std == 0 or np.isnan(std) or np.isinf(std):
        return s - s.mean()
    return (s - s.mean()) / std

def _last_trading_day(db_path: str, benchmark: str = "^TWII") -> str:
    """Find the latest trading date in DB."""
    d = get_last_date(db_path, benchmark)
    if d is None:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
        d = row[0] if row else None
    if d is None:
        raise RuntimeError("No trading dates found in DB. Please run download_data first.")
    return d

def _load_signal_contexts(
    cfg: Config,
    rebal_date: pd.Timestamp,
    symbols: list[str],
) -> list[tuple[str, pd.DataFrame, pd.Series, pd.Series]]:
    rebal_str = rebal_date.strftime("%Y-%m-%d")
    lookback_start = (
        rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)
    ).strftime("%Y-%m-%d")
    
    # We only need 1 day of prediction target for formatting
    y_ts = pd.Series([rebal_date])

    rows = query_symbols_window(
        cfg.db_path,
        symbols,
        start=lookback_start,
        end=rebal_str,
    )
    if rows.empty:
        return []

    grouped = {sym: grp.reset_index(drop=True) for sym, grp in rows.groupby("symbol", sort=False)}
    contexts = []
    for sym in symbols:
        df = grouped.get(sym)
        if df is None or len(df) < cfg.lookback_window:
            continue
        ctx = df.iloc[-cfg.lookback_window:].reset_index(drop=True)
        ctx_df = ctx[_PRICE_COLUMNS].reset_index(drop=True)
        if ctx_df.isnull().any().any():
            continue
        x_ts = pd.to_datetime(ctx["date"]).reset_index(drop=True)
        contexts.append((sym, ctx_df, x_ts, y_ts.copy()))
    return contexts

def get_ensemble_signals_for_date(
    predictor,
    booster_full: xgb.Booster,
    booster_raw: xgb.Booster,
    feat_cols_full: list[str],
    feat_cols_raw: list[str],
    weight: float,
    cfg: Config,
    rebal_date: pd.Timestamp,
    symbols: list[str],
    xreg_enabled: bool = False,
    xreg_mult: float = 2.0,
    xreg_lookback: int = 60,
    xreg_purging_gap: int = 5,
    hold_days: int = 5,
) -> dict[str, float]:
    """Execute ensemble inference for a single date."""
    contexts = _load_signal_contexts(cfg, rebal_date, symbols)
    batch_syms = [sym for sym, _, _, _ in contexts]
    batch_dfs = [ctx_df for _, ctx_df, _, _ in contexts]
    batch_xts = [x_ts for _, _, x_ts, _ in contexts]
    batch_yts = [y_ts for _, _, _, y_ts in contexts]

    print(f"  Extracting features and predicting for {len(batch_syms)} active symbols...")
    sys.stdout.flush()

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

    signals: dict[str, float] = {}
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
            signals[sym] = float(pred)

        # Apply XReg adjustment if enabled
        if xreg_enabled:
            from finetune_tw.trading_calendar import twse_trading_days
            from finetune_tw.xreg import apply_xreg_adjustment
            trading_days = sorted(list(twse_trading_days(cfg.db_path)))
            signals = apply_xreg_adjustment(
                db_path=cfg.db_path,
                symbols=ordered_syms,
                rebal_date_str=rebal_date.strftime("%Y-%m-%d"),
                scores_gbdt=signals,
                trading_days=trading_days,
                lookback=xreg_lookback,
                purging_gap=xreg_purging_gap,
                hold_days=hold_days,
                mult=xreg_mult,
            )

    return signals

def main() -> None:
    parser = argparse.ArgumentParser(description="Output today's Kronos stock signals using the static Z-Score ensemble blending model.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--xgb_model_full", required=True, help="XGBoost Full model file path")
    parser.add_argument("--xgb_model_raw", required=True, help="XGBoost Raw model file path")
    parser.add_argument("--weight", type=float, default=0.6, help="Weight for full model (w * z_full + (1-w) * z_raw)")
    parser.add_argument("--model", default="pretrained", help="Backbone model key")
    parser.add_argument("--date", default=None, help="Specify trading day (YYYY-MM-DD), default is latest in DB")
    parser.add_argument("--top_k", type=int, default=None, help="Number of holdings (default is config.top_k)")
    parser.add_argument("--holdings", default="", help="Currently held stock symbols, comma separated (for rebalance recommendations)")
    parser.add_argument("--xreg_enabled", action="store_true", help="Enable Exogenous Residual Regression (XReg) adjustment")
    parser.add_argument("--xreg_mult", type=float, default=2.0, help="Multiplier for XReg score adjustments")
    parser.add_argument("--xreg_lookback", type=int, default=60, help="Lookback window in trading days for fitting Ridge")
    parser.add_argument("--xreg_purging_gap", type=int, default=5, help="Purging gap in days to avoid look-ahead leak")
    parser.add_argument("--xreg_alpha", type=float, default=1.0, help="Ridge regression L2 regularization alpha")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    top_k = args.top_k or cfg.top_k

    # Determine signal date
    if args.date:
        rebal_date = pd.Timestamp(args.date)
    else:
        latest = _last_trading_day(cfg.db_path, cfg.benchmark_symbol)
        rebal_date = pd.Timestamp(latest)

    print(f"\n=== Kronos Ensemble Blending daily signals ===")
    print(f"  Backbone : {args.model}")
    print(f"  Full Model: {Path(args.xgb_model_full).name}")
    print(f"  Raw Model : {Path(args.xgb_model_raw).name}")
    print(f"  Weight w  : {args.weight:.2f} (Full) / {1.0 - args.weight:.2f} (Raw)")
    print(f"  Signal Date: {rebal_date.date()}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if args.xreg_enabled:
        print(f"  XReg Enabled: mult={args.xreg_mult}, lookback={args.xreg_lookback}, purging_gap={args.xreg_purging_gap}")
    print()

    # Restore XGB checkpoints from HF when the local files are missing
    _ensure_xgb_model(args.xgb_model_full)
    _ensure_xgb_model(args.xgb_model_raw)

    # Load feature metadata from sidecars
    feat_cols_full = _load_model_feature_columns(args.xgb_model_full)
    feat_cols_raw = _load_model_feature_columns(args.xgb_model_raw)
    if feat_cols_full is None or feat_cols_raw is None:
        print("Error: Missing summary sidecar files (.summary.json) for feature column matching.")
        sys.exit(1)

    # Load models
    specs = build_model_specs(cfg)
    if "pretrained" in specs:
        specs["pretrained"].tok_kwargs["revision"] = "0e0117387f39004a9016484a186a908917e22426"
        specs["pretrained"].pred_kwargs["revision"] = "2b554741eca47781b64468546e77fef3e85130e6"
    predictor = load_predictor_from_spec(specs[args.model], cfg)
    
    booster_full = xgb.Booster()
    booster_full.load_model(args.xgb_model_full)
    
    booster_raw = xgb.Booster()
    booster_raw.load_model(args.xgb_model_raw)

    # Load symbols
    all_symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]

    # Run inference
    signals = get_ensemble_signals_for_date(
        predictor=predictor,
        booster_full=booster_full,
        booster_raw=booster_raw,
        feat_cols_full=feat_cols_full,
        feat_cols_raw=feat_cols_raw,
        weight=args.weight,
        cfg=cfg,
        rebal_date=rebal_date,
        symbols=all_symbols,
        xreg_enabled=args.xreg_enabled,
        xreg_mult=args.xreg_mult,
        xreg_lookback=args.xreg_lookback,
        xreg_purging_gap=args.xreg_purging_gap,
        hold_days=cfg.hold_days,
    )

    if not signals:
        print("Warning: No signals generated. Please ensure database is updated.")
        sys.exit(1)

    # Rank and select Top-k
    top_set = rank_stocks(signals, top_k=top_k, threshold=cfg.min_signal_threshold)
    ranked = sorted(top_set, key=lambda s: signals[s], reverse=True)

    print(f"\n【Ensemble Blending Selection Results】(Date: {rebal_date.date()})")
    print(f"{'Rank':>4}  {'Symbol':>8}  {'Blended Z-Score':>16}")
    print("-" * 36)
    for rank, sym in enumerate(ranked, 1):
        print(f"  {rank:>2}   {sym:>8}   {signals[sym]:>+16.4f}")

    # Rebalance Suggestions
    if args.holdings:
        current = set(args.holdings.split(","))
        to_sell = current - top_set
        to_buy = top_set - current
        hold = current & top_set
        print(f"\n【Rebalance Advice】(Current Portfolio: {sorted(current)})")
        if hold:
            print(f"  Hold: {sorted(hold)}")
        if to_sell:
            print(f"  Sell: {sorted(to_sell)}")
        if to_buy:
            print(f"  Buy : {sorted(to_buy)}")
        if not to_sell and not to_buy:
            print("  Portfolio is aligned. No action needed.")
            
    print()

if __name__ == "__main__":
    main()
