"""Extract frozen Kronos last-layer hidden states as embeddings for XGBoost ranking (Round 6 / M1).

Kronos is never updated here — this only runs forward passes under torch.no_grad().

Batch 3 production flow should use enrich_round6_features.py rather than this script. This script still
finishes with a whole-frame late-stage cross-sectional rank pass via add_cross_sectional_rank_features().
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from finetune_tw.backtest import build_model_specs, load_predictor_from_spec
from finetune_tw.backtest_data import build_rebalance_inputs, load_symbol_history_frames
from finetune_tw.config import Config
from finetune_tw.db import list_symbols
from finetune_tw.feature_engineering import add_cross_sectional_rank_features, compute_technical_features
from finetune_tw.trading_calendar import twse_trading_days

BATCH_SIZE = 64


def extract_embeddings_batch(
    predictor,
    df_list: list[pd.DataFrame],
    x_timestamp_list: list[pd.Series],
    layer_indices: list[int] | None = None,
) -> np.ndarray:
    """Mean-pool the frozen Kronos transformer's last-layer hidden state over the lookback window.

    Reuses KronosPredictor.prepare_batch_inputs for normalization (lookback-only mean/std + clip,
    identical to predict()/predict_batch()), then tokenizer.encode + model.decode_s1 to obtain the
    per-timestep context (B, seq_len, d_model), mean-pooled over seq_len -> (B, d_model).
    """
    if not df_list:
        return np.empty((0, predictor.model.d_model), dtype=np.float32)

    dummy_y_ts = [pd.Series(pd.bdate_range(x_ts.iloc[-1] + pd.Timedelta(days=1), periods=1))
                  for x_ts in x_timestamp_list]

    x_batch, x_stamp_batch, _, _, _, _ = predictor.prepare_batch_inputs(
        df_list=df_list,
        x_timestamp_list=x_timestamp_list,
        y_timestamp_list=dummy_y_ts,
        pred_len=1,
    )

    device = predictor.device
    x_tensor = torch.from_numpy(x_batch).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp_batch).to(device)

    with torch.no_grad():
        s1_ids, s2_ids = predictor.tokenizer.encode(x_tensor, half=True)
        model = predictor.model

        if layer_indices is None:
            _, context = model.decode_s1(s1_ids, s2_ids, x_stamp_tensor)
            pooled = context.mean(dim=1)
        else:
            x = model.embedding([s1_ids, s2_ids])
            x = x + model.time_emb(x_stamp_tensor)
            x = model.token_drop(x)
            layer_outputs = []
            for i, layer in enumerate(model.transformer):
                x = layer(x)
                if i in layer_indices:
                    layer_outputs.append(x.mean(dim=1))
            pooled = torch.cat(layer_outputs, dim=1)

    return pooled.cpu().numpy().astype(np.float32)


def _realized_open_to_open_labels(
    price_frames: dict[str, pd.DataFrame],
    symbols: list[str],
    rebal_date: pd.Timestamp,
    horizon: int,
) -> dict[str, float]:
    """label[sym] = open[t+horizon+1] / open[t+1] - 1, matching backtest_next_open's signal exactly."""
    labels: dict[str, float] = {}
    for sym in symbols:
        frame = price_frames.get(sym)
        if frame is None:
            continue
        future = frame.loc[frame.index > rebal_date, "open"]
        if len(future) <= horizon:
            continue
        t1 = future.iloc[0]
        th = future.iloc[horizon]
        if t1 == 0:
            continue
        labels[sym] = float(th / t1 - 1.0)
    return labels


def build_embedding_dataset(
    cfg: Config,
    predictor,
    symbols: list[str],
    rebal_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    """One row per (date, symbol): embedding + realized open-to-open label at `horizon` days."""
    if len(rebal_dates) == 0:
        return pd.DataFrame()

    preload_start = (
        rebal_dates.min() - pd.Timedelta(days=cfg.lookback_window * 2)
    ).strftime("%Y-%m-%d")
    preload_end = (
        rebal_dates.max() + pd.Timedelta(days=horizon * 3)
    ).strftime("%Y-%m-%d")
    history_frames = load_symbol_history_frames(cfg.db_path, symbols, start=preload_start, end=preload_end)

    rows: list[dict] = []
    for i, rebal_date in enumerate(rebal_dates):
        cutoff = rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)
        recent = {sym: frame.loc[frame.index >= cutoff] for sym, frame in history_frames.items()}
        batch_syms, batch_dfs, batch_xts, _ = build_rebalance_inputs(
            recent, symbols, rebal_date, cfg.lookback_window, pred_len=1,
        )
        labels = _realized_open_to_open_labels(history_frames, batch_syms, rebal_date, horizon)

        keep_idx = [j for j, sym in enumerate(batch_syms) if sym in labels]
        if not keep_idx:
            continue
        kept_syms = [batch_syms[j] for j in keep_idx]
        kept_dfs = [batch_dfs[j] for j in keep_idx]
        kept_xts = [batch_xts[j] for j in keep_idx]

        for b in range(0, len(kept_syms), BATCH_SIZE):
            sub_syms = kept_syms[b:b + BATCH_SIZE]
            sub_dfs = kept_dfs[b:b + BATCH_SIZE]
            sub_xts = kept_xts[b:b + BATCH_SIZE]
            embeddings = extract_embeddings_batch(predictor, sub_dfs, sub_xts)
            for sym, emb, ctx_df in zip(sub_syms, embeddings, sub_dfs):
                row = {"date": rebal_date.strftime("%Y-%m-%d"), "symbol": sym, "label": labels[sym]}
                row.update({f"emb_{k}": float(v) for k, v in enumerate(emb)})
                row.update(compute_technical_features(ctx_df))
                rows.append(row)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(rebal_dates)}] {rebal_date.date()}: {len(keep_idx)} symbols")
            sys.stdout.flush()

    return add_cross_sectional_rank_features(pd.DataFrame(rows))


def _select_symbols(symbols: list[str], max_symbols: int | None) -> list[str]:
    """Deterministically truncate the (already-sorted) symbol universe for small-scale local runs."""
    if max_symbols is None:
        return symbols
    return symbols[:max_symbols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--model", default="pretrained", choices=list(build_model_specs(Config()).keys()))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--max-symbols", type=int, default=None,
        help="Truncate the symbol universe to the first N (sorted by ticker) for quick local CPU sampling.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    specs = build_model_specs(cfg)
    predictor = load_predictor_from_spec(specs[args.model], cfg)

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    symbols = _select_symbols(symbols, args.max_symbols)
    all_trading_days = twse_trading_days(cfg.db_path)
    rebal_dates = pd.DatetimeIndex(
        [pd.Timestamp(day) for day in sorted(all_trading_days) if args.start <= day <= args.end]
    )

    df = build_embedding_dataset(cfg, predictor, symbols, rebal_dates, args.horizon)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
