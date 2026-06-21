"""
Forecast-accuracy evaluation, decoupled from the portfolio backtest.

Answers "are the model's predictions close to reality?" directly, separate from
any holdings / position-sizing logic:

  1. val_loss  — recomputed on best_model (token-space CE, the training metric)
  2. fidelity  — per-horizon MAPE / RMSE / direction hit-rate on predicted close
                 vs actual close, compared against a no-change naive baseline
  3. alpha     — per-horizon Information Coefficient (Spearman rank corr between
                 predicted return and realised return), the bridge to the backtest
  4. overlays  — multi-sample predicted vs actual close curves for a few symbols

Run:
    python -m finetune_tw.eval_forecast --config <cfg>
    python -m finetune_tw.eval_forecast --config <cfg> --smoke   # quick self-test
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.config import Config
from finetune_tw.db import query_symbol, list_symbols
from finetune_tw.dataset import MultiStockDataset
from finetune_tw.hf_utils import resolve_src
from finetune_tw.ic_validation import rank_ic as _rank_ic
from finetune_tw.train_predictor import _validate_predictor, _resolve_amp


def _safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


# ── Stage 1: recompute val_loss ─────────────────────────────────────────────

def compute_val_loss(cfg, tokenizer, model, device) -> float:
    val_ds = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                               cfg.train_end_date, cfg.val_end_date, cfg.clip, cfg.seed + 1)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    amp_enabled, amp_dtype = _resolve_amp(cfg.amp_dtype)
    amp_enabled = amp_enabled and device.type == "cuda"
    return _validate_predictor(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)


# ── Forecast sweep (shared by fidelity + alpha) ─────────────────────────────

def _build_batch(cfg, symbols, rebal_date):
    """Mirror backtest.py context construction. Returns parallel lists."""
    rebal_str = rebal_date.strftime("%Y-%m-%d")
    lookback_start = (rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
    y_ts = pd.date_range(rebal_date, periods=cfg.pred_len, freq="B")
    syms, dfs, xts, yts, ctx_last_dates = [], [], [], [], []
    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=lookback_start, end=rebal_str)
        if len(df) < cfg.lookback_window:
            continue
        ctx = df.iloc[-cfg.lookback_window:]
        ctx_df = ctx[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        if ctx_df.isnull().any().any():
            continue
        xs = pd.to_datetime(ctx["date"]).reset_index(drop=True)
        syms.append(sym)
        dfs.append(ctx_df)
        xts.append(xs)
        yts.append(pd.Series(y_ts))
        ctx_last_dates.append(xs.iloc[-1])
    return syms, dfs, xts, yts, ctx_last_dates


def run_eval(cfg, smoke: bool = False, fidelity_symbols: int = 12,
             fidelity_samples: int = 20, batch_size: int = 64,
             baseline: bool = False) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if baseline:
        tok_src, tok_kw   = cfg.pretrained_tokenizer, {}
        pred_src, pred_kw = cfg.pretrained_predictor, {}
        tag = "baseline"
    else:
        exp_dir   = Path(cfg.output_dir) / cfg.exp_name
        tok_src,  tok_kw  = resolve_src(exp_dir / "tokenizer" / "best_model",
                                        cfg.hf_repo, "tokenizer/best_model", cfg.hf_revision)
        pred_src, pred_kw = resolve_src(exp_dir / "predictor" / "best_model",
                                        cfg.hf_repo, "predictor/best_model", cfg.hf_revision)
        tag = "finetuned"
    print(f"loading {tag}: tokenizer={tok_src} predictor={pred_src}", flush=True)

    tokenizer = KronosTokenizer.from_pretrained(tok_src, **tok_kw).to(device)
    model = Kronos.from_pretrained(pred_src, **pred_kw).to(device)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
    tokenizer.eval(); model.eval()

    result: dict = {"variant": tag, "config": {
        "lookback_window": cfg.lookback_window, "pred_len": cfg.pred_len,
        "hold_days": cfg.hold_days, "test_start": cfg.test_start_date,
    }}

    # Stage 1: val_loss
    if not smoke:
        print("[1/3] recomputing val_loss ...", flush=True)
        result["val_loss"] = compute_val_loss(cfg, tokenizer, model, device)
        print(f"      val_loss = {result['val_loss']:.4f}", flush=True)

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    if smoke:
        symbols = symbols[:40]
    test_end = str(pd.Timestamp.today().date())

    # Preload actual OHLC over test period (+buffer) for alignment
    buffer_start = (pd.Timestamp(cfg.test_start_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    actual: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=buffer_start, end=test_end)
        if len(df) > 0:
            df = df.copy()
            df.index = pd.DatetimeIndex(df["date"])
            actual[sym] = df

    all_dates = pd.bdate_range(cfg.test_start_date, test_end)
    rebalance_dates = all_dates[::cfg.hold_days]
    if smoke:
        rebalance_dates = rebalance_dates[:2]

    L = cfg.pred_len
    # accumulators per horizon h=0..L-1
    rec = {h: {"pred": [], "act": [], "ctx": [], "date": []} for h in range(L)}

    print(f"[2/3] forecast sweep: {len(symbols)} symbols x {len(rebalance_dates)} dates "
          f"(sample_count=1)", flush=True)
    for i, rebal_date in enumerate(rebalance_dates):
        syms, dfs, xts, yts, ctx_last_dates = _build_batch(cfg, symbols, rebal_date)
        with torch.no_grad():
            for b in range(0, len(syms), batch_size):
                preds = predictor.predict_batch(
                    df_list=dfs[b:b + batch_size],
                    x_timestamp_list=xts[b:b + batch_size],
                    y_timestamp_list=yts[b:b + batch_size],
                    pred_len=L, T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False,
                )
                for sym, pred, ctx_df, cld in zip(syms[b:b + batch_size], preds,
                                                  dfs[b:b + batch_size], ctx_last_dates[b:b + batch_size]):
                    if pred is None or len(pred) < L:
                        continue
                    ctx_close = float(ctx_df["close"].iloc[-1])
                    pred_close = pred["close"].values.astype(float)
                    ser = actual.get(sym)
                    if ser is None:
                        continue
                    pos = ser.index.searchsorted(cld, side="right")  # first row strictly after ctx end
                    fut = ser.iloc[pos:pos + L]
                    act_close = fut["close"].values.astype(float)
                    for h in range(min(L, len(act_close))):
                        rec[h]["pred"].append(pred_close[h])
                        rec[h]["act"].append(act_close[h])
                        rec[h]["ctx"].append(ctx_close)
                        rec[h]["date"].append(rebal_date)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"      [{i+1}/{len(rebalance_dates)}] {rebal_date.date()}", flush=True)

    # Aggregate per-horizon metrics
    horizons = []
    for h in range(L):
        p = np.array(rec[h]["pred"]); a = np.array(rec[h]["act"]); c = np.array(rec[h]["ctx"])
        d = np.array(rec[h]["date"])
        if len(p) == 0:
            continue
        pred_ret = p / c - 1.0
        act_ret = a / c - 1.0
        mape = _safe_mean(np.abs(p - a) / np.abs(a))
        mape_base = _safe_mean(np.abs(c - a) / np.abs(a))  # naive no-change
        rmse_ret = float(np.sqrt(_safe_mean((pred_ret - act_ret) ** 2)))
        nz = (np.abs(pred_ret) > 1e-9) & (np.abs(act_ret) > 1e-9)
        direction = _safe_mean((np.sign(pred_ret[nz]) == np.sign(act_ret[nz])).astype(float))
        up_rate = _safe_mean((act_ret > 0).astype(float))  # baseline: always-up hit-rate
        # IC per rebalance date then average
        ics = []
        for dd in pd.unique(d):
            mask = d == dd
            ics.append(_rank_ic(pred_ret[mask], act_ret[mask]))
        ics = np.array([x for x in ics if np.isfinite(x)])
        ic_mean = float(ics.mean()) if len(ics) else float("nan")
        ic_std = float(ics.std()) if len(ics) else float("nan")
        horizons.append({
            "h": h + 1, "n": int(len(p)),
            "mape": mape, "mape_naive": mape_base,
            "rmse_ret": rmse_ret,
            "direction_acc": direction, "always_up_rate": up_rate,
            "ic_mean": ic_mean, "ic_std": ic_std,
            "ic_ir": (ic_mean / ic_std) if ic_std and np.isfinite(ic_std) and ic_std > 1e-9 else float("nan"),
            "ic_positive_rate": float((ics > 0).mean()) if len(ics) else float("nan"),
        })
    result["horizons"] = horizons

    # Stage 3: multi-sample fidelity overlays
    print(f"[3/3] fidelity overlays: {fidelity_symbols} symbols, "
          f"sample_count={fidelity_samples}", flush=True)
    fid_date = rebalance_dates[len(rebalance_dates) // 2]
    syms, dfs, xts, yts, ctx_last_dates = _build_batch(cfg, symbols, fid_date)
    keep = []
    for j, sym in enumerate(syms):
        ser = actual.get(sym)
        if ser is None:
            continue
        pos = ser.index.searchsorted(ctx_last_dates[j], side="right")
        if len(ser.iloc[pos:pos + L]) >= L:
            keep.append(j)
        if len(keep) >= fidelity_symbols:
            break
    overlays = []
    if keep:
        with torch.no_grad():
            preds = predictor.predict_batch(
                df_list=[dfs[j] for j in keep],
                x_timestamp_list=[xts[j] for j in keep],
                y_timestamp_list=[yts[j] for j in keep],
                pred_len=L, T=1.0, top_k=0, top_p=0.9,
                sample_count=fidelity_samples, verbose=False,
            )
        for k, j in enumerate(keep):
            ser = actual[syms[j]]
            pos = ser.index.searchsorted(ctx_last_dates[j], side="right")
            act_close = ser.iloc[pos:pos + L]["close"].values.astype(float)
            overlays.append({"symbol": syms[j],
                             "pred_close": preds[k]["close"].values.astype(float).tolist(),
                             "actual_close": act_close.tolist()})
        n = len(overlays)
        cols = 3; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, ov in zip(axes, overlays):
            ax.plot(range(1, L + 1), ov["actual_close"], "o-", label="actual", color="C0")
            ax.plot(range(1, L + 1), ov["pred_close"], "s--", label="pred", color="C1")
            ax.set_title(ov["symbol"], fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes[len(overlays):]:
            ax.axis("off")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Predicted vs Actual close — {fid_date.date()} "
                     f"(sample_count={fidelity_samples})")
        fig.tight_layout()
        out_dir = Path(cfg.output_dir) / cfg.exp_name / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"eval_fidelity_{tag}.png", dpi=110)
        print(f"      saved {out_dir / f'eval_fidelity_{tag}.png'}", flush=True)
    result["fidelity_date"] = str(fid_date.date())
    result["overlays"] = overlays

    out_dir = Path(cfg.output_dir) / cfg.exp_name / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_metrics_{tag}.json").write_text(json.dumps(result, indent=2))
    print(f"saved {out_dir / f'eval_metrics_{tag}.json'}", flush=True)
    return result


def _print_summary(r: dict) -> None:
    print("\n=== Forecast Evaluation Summary ===")
    if "val_loss" in r:
        print(f"val_loss (token CE): {r['val_loss']:.4f}")
    print(f"{'h':>3} {'n':>7} {'MAPE':>7} {'MAPE0':>7} {'dir%':>6} {'IC':>7} {'IC_IR':>6} {'IC>0%':>6}")
    for row in r.get("horizons", []):
        print(f"{row['h']:>3} {row['n']:>7} {row['mape']:>7.4f} {row['mape_naive']:>7.4f} "
              f"{row['direction_acc']*100:>5.1f} {row['ic_mean']:>7.4f} {row['ic_ir']:>6.2f} "
              f"{row['ic_positive_rate']*100:>5.1f}")
    print("MAPE0 = naive no-change baseline; dir% = direction hit-rate; "
          "IC = cross-sectional Spearman.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fidelity_symbols", type=int, default=12)
    parser.add_argument("--fidelity_samples", type=int, default=20)
    parser.add_argument("--baseline", action="store_true",
                        help="evaluate un-fine-tuned pretrained Kronos as control group")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    r = run_eval(cfg, smoke=args.smoke, batch_size=args.batch_size,
                 fidelity_symbols=args.fidelity_symbols, fidelity_samples=args.fidelity_samples,
                 baseline=args.baseline)
    _print_summary(r)


if __name__ == "__main__":
    main()
