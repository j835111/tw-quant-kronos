"""
Run inference for one model, output daily-returns JSON for later plotting.

Usage:
    python -m finetune_tw.backtest \\
        --config finetune_tw/configs/config_tw_daily_rtx6000.yaml \\
        --model round2

    # Custom hold variants:
    python -m finetune_tw.backtest --config ... --model round2 --hold_days_list 5 10 20

Model choices: pretrained | round0 | round1 | round2
Output: {output_dir}/{exp_name}/backtest_returns_{model}.json
         {output_dir}/{exp_name}/backtest_{model}.png
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from model import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.config import Config
from finetune_tw.db import query_symbol, list_symbols
from finetune_tw.hf_utils import has_weights


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(daily_returns: pd.Series) -> dict:
    ann_ret = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
    cum = (1 + daily_returns).cumprod()
    max_dd = ((cum.cummax() - cum) / cum.cummax()).max()
    return {"annualised_return": float(ann_ret), "sharpe": float(sharpe),
            "max_drawdown": float(max_dd)}


# ── Portfolio ─────────────────────────────────────────────────────────────────

def rank_stocks(signals: dict[str, float], top_k: int, threshold: float = 0.0) -> set[str]:
    eligible = {sym: v for sym, v in signals.items() if v >= threshold}
    return set(sorted(eligible, key=eligible.__getitem__, reverse=True)[:top_k])


def build_portfolio_returns(
    price_data: dict[str, pd.Series],
    holdings_sequence: list[set[str]],
    rebalance_dates: pd.Index,
    weights: dict | None = None,
) -> tuple[pd.Series, pd.Series]:
    def resolve_period_weights(rebalance_date) -> dict[str, float] | None:
        if not weights:
            return None
        sample = next(iter(weights.values()), None)
        if isinstance(sample, Mapping):
            date_key = pd.Timestamp(rebalance_date)
            period_weights = weights.get(date_key)
            if period_weights is None:
                period_weights = weights.get(date_key.strftime("%Y-%m-%d"))
            return dict(period_weights) if period_weights else None
        return dict(weights)

    all_daily: list[pd.Series] = []
    period_rets: list[float] = []
    period_dates: list = []
    for i in range(len(rebalance_dates) - 1):
        date, next_date = rebalance_dates[i], rebalance_dates[i + 1]
        period_weights = resolve_period_weights(date)
        daily_sym_series = []
        sym_period_rets: dict[str, float] = {}
        for sym in holdings_sequence[i]:
            if sym not in price_data:
                continue
            series = price_data[sym]
            sub = series[(series.index >= date) & (series.index <= next_date)]
            if len(sub) >= 2:
                daily_sym_series.append(sub.pct_change().dropna().rename(sym))
                sym_period_rets[sym] = float(sub.iloc[-1] / sub.iloc[0] - 1.0)
        if daily_sym_series:
            daily_frame = pd.concat(daily_sym_series, axis=1)
            if period_weights is None:
                all_daily.append(daily_frame.mean(axis=1))
            else:
                weight_series = pd.Series(
                    {sym: float(period_weights.get(sym, 0.0)) for sym in daily_frame.columns},
                    dtype=float,
                )
                denom = daily_frame.notna().mul(weight_series, axis=1).sum(axis=1)
                numer = daily_frame.mul(weight_series, axis=1).sum(axis=1, min_count=1)
                weighted_daily = numer[denom > 0] / denom[denom > 0]
                all_daily.append(weighted_daily)
        if sym_period_rets:
            if period_weights is None:
                period_rets.append(float(np.mean(list(sym_period_rets.values()))))
            else:
                total_weight = float(sum(period_weights.get(sym, 0.0) for sym in sym_period_rets))
                if total_weight > 0:
                    weighted_ret = sum(
                        sym_period_rets[sym] * float(period_weights.get(sym, 0.0))
                        for sym in sym_period_rets
                    ) / total_weight
                    period_rets.append(float(weighted_ret))
                else:
                    period_rets.append(0.0)
        else:
            period_rets.append(0.0)
        period_dates.append(date)

    period_returns = pd.Series(period_rets, index=pd.DatetimeIndex(period_dates))
    if not all_daily:
        return period_returns, pd.Series(dtype=float)
    dr = pd.concat(all_daily).sort_index()
    daily_returns = dr[~dr.index.duplicated(keep="first")]
    return period_returns, daily_returns


def signals_to_holdings(
    raw_preds: dict[str, dict[str, pd.Series]],
    rebal_dates: pd.DatetimeIndex,
    hold_days: int,
    top_k: int,
    threshold: float = 0.0,
) -> list[set[str]]:
    holdings = []
    for d in rebal_dates:
        date_preds = raw_preds.get(d.strftime("%Y-%m-%d"), {})
        signals = {sym: ret.iloc[hold_days - 1]
                   for sym, ret in date_preds.items() if len(ret) >= hold_days}
        holdings.append(rank_stocks(signals, top_k, threshold))
    return holdings


# ── Model spec ────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    label: str
    tok_src: str
    tok_kwargs: dict
    pred_src: str
    pred_kwargs: dict


def _hf_local(repo_id: str, subfolder: str, revision: str) -> str:
    """Download a HF repo subfolder to cache and return the local path."""
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    local = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[f"{subfolder}/*"],
        token=token,
    )
    return str(Path(local) / subfolder)


def build_model_specs(cfg: Config) -> dict[str, ModelSpec]:
    exp_dir = Path(cfg.output_dir) / cfg.exp_name
    local_tok  = exp_dir / "tokenizer" / "best_model"
    local_pred = exp_dir / "predictor" / "best_model"
    hf_repo = "j835111/kronos-tw-finetune"

    # Round 0/1 share the same tokenizer (local if available, else HF round-0)
    if has_weights(local_tok):
        tok_src_local = str(local_tok)
    else:
        tok_src_local = None  # resolved lazily in load_predictor_from_spec

    return {
        "pretrained": ModelSpec(
            label="Pretrained",
            tok_src=cfg.pretrained_tokenizer, tok_kwargs={},
            pred_src=cfg.pretrained_predictor, pred_kwargs={},
        ),
        "round0": ModelSpec(
            label="Round 0",
            tok_src=tok_src_local or f"hf://{hf_repo}@round-0/tokenizer/best_model",
            tok_kwargs={},
            pred_src=f"hf://{hf_repo}@round-0/predictor/best_model",
            pred_kwargs={},
        ),
        "round1": ModelSpec(
            label="Round 1",
            tok_src=tok_src_local or f"hf://{hf_repo}@round-0/tokenizer/best_model",
            tok_kwargs={},
            pred_src=str(local_pred) if has_weights(local_pred)
                     else f"hf://{hf_repo}@round-1/predictor/best_model",
            pred_kwargs={},
        ),
        "round2": ModelSpec(
            label="Round 2",
            tok_src=tok_src_local or f"hf://{hf_repo}@round-0/tokenizer/best_model",
            tok_kwargs={},
            pred_src=str(local_pred) if has_weights(local_pred)
                     else f"hf://{hf_repo}@round-2/predictor/best_model",
            pred_kwargs={},
        ),
    }


def _resolve_src(src: str) -> str:
    """If src is hf://repo_id@revision/subfolder, download and return local path.
    repo_id may contain '/' (e.g. owner/repo), so split on '@' first."""
    if not src.startswith("hf://"):
        return src
    # hf://owner/repo@revision/subfolder
    rest = src[5:]                              # "owner/repo@rev/sub/folder"
    repo_id, rev_and_sub = rest.split("@", 1)  # "owner/repo", "rev/sub/folder"
    revision, subfolder = rev_and_sub.split("/", 1)  # "rev", "sub/folder"
    print(f"  [hf] downloading {repo_id}@{revision}/{subfolder} ...")
    sys.stdout.flush()
    return _hf_local(repo_id, subfolder, revision)


def load_predictor_from_spec(spec: ModelSpec, cfg: Config) -> KronosPredictor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok_path  = _resolve_src(spec.tok_src)
    pred_path = _resolve_src(spec.pred_src)
    print(f"  tokenizer  ← {tok_path}")
    print(f"  predictor  ← {pred_path}")
    sys.stdout.flush()
    tokenizer = KronosTokenizer.from_pretrained(tok_path, **spec.tok_kwargs)
    model = Kronos.from_pretrained(pred_path, **spec.pred_kwargs)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
    tokenizer.eval(); model.eval()
    return predictor


# ── Inference ─────────────────────────────────────────────────────────────────

def compute_raw_signals(
    predictor: KronosPredictor,
    cfg: Config,
    rebal_dates: pd.DatetimeIndex,
    pred_len: int,
    symbols: list[str],
) -> dict[str, dict[str, pd.Series]]:
    """raw_preds[date_str][sym] = Series of predicted close returns, iloc[h-1] = h-day return."""
    BATCH_SIZE = 64
    raw_preds: dict[str, dict[str, pd.Series]] = {}

    for i, rebal_date in enumerate(rebal_dates):
        rebal_str = rebal_date.strftime("%Y-%m-%d")
        lookback_start = (rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
        y_ts = pd.date_range(rebal_date, periods=pred_len, freq="B")

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

        date_preds: dict[str, pd.Series] = {}
        with torch.no_grad():
            for b in range(0, len(batch_syms), BATCH_SIZE):
                preds = predictor.predict_batch(
                    df_list=batch_dfs[b:b + BATCH_SIZE],
                    x_timestamp_list=batch_xts[b:b + BATCH_SIZE],
                    y_timestamp_list=batch_yts[b:b + BATCH_SIZE],
                    pred_len=pred_len,
                    T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False,
                )
                for sym, pred, ctx_df in zip(batch_syms[b:b + BATCH_SIZE], preds,
                                              batch_dfs[b:b + BATCH_SIZE]):
                    if pred is not None and len(pred) >= pred_len:
                        last_close = ctx_df["close"].iloc[-1]
                        date_preds[sym] = pred["close"].reset_index(drop=True) / last_close - 1

        raw_preds[rebal_str] = date_preds
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(rebal_dates)}] {rebal_str}: {len(date_preds)} signals")
            sys.stdout.flush()

    return raw_preds


# ── Plotting ──────────────────────────────────────────────────────────────────

_HOLD_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]
_BM_COLOR    = "#9E9E9E"
_DD_ALPHA    = 0.18


def plot_backtest_results(data: dict, out_dir: Path) -> Path:
    """Generate a 2×2 chart grid from backtest JSON and save as PNG.

    Layout:
        Top-left:    Cumulative returns — all hold variants + benchmark
        Top-right:   Per-variant Sharpe / Annual Return / Max DD bar chart
        Bottom row:  One drawdown chart per hold variant (up to 3)
    """
    hold_variants = data["hold_variants"]
    bm = data["benchmark"]
    model_label = data["model_label"]
    hold_keys = sorted(hold_variants, key=int)

    bm_dates   = pd.DatetimeIndex(bm["dates"])
    bm_cum     = (1 + pd.Series(bm["daily_returns"], index=bm_dates)).cumprod()

    n_holds = len(hold_keys)
    colors  = (_HOLD_COLORS * 4)[:n_holds]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Backtest — {model_label}  ({data['test_start']} → {data['test_end']}  top-K={data['top_k']})",
                 fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(2, max(n_holds, 2), hspace=0.38, wspace=0.32)

    # ── Top-left: cumulative returns ──────────────────────────────────────────
    ax_cum = fig.add_subplot(gs[0, :n_holds // 2 + 1])
    ax_cum.plot(bm_cum.index, bm_cum.values, color=_BM_COLOR, lw=1.5,
                linestyle="--", label=f"^TWII  Sharpe={bm['metrics']['sharpe']:.2f}")
    for hk, col in zip(hold_keys, colors):
        v = hold_variants[hk]
        dr = pd.Series(v["daily_returns"], index=pd.DatetimeIndex(v["dates"]))
        cum = (1 + dr).cumprod()
        m = v["metrics"]
        ax_cum.plot(cum.index, cum.values, color=col, lw=1.8,
                    label=f"hold={hk}d  Sharpe={m['sharpe']:.2f}  Ann={m['annualised_return']:.1%}")
    ax_cum.set_ylabel("Cumulative Return")
    ax_cum.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax_cum.axhline(1, color="black", lw=0.6, ls=":")
    ax_cum.legend(fontsize=7.5, loc="upper left")
    ax_cum.set_title("Cumulative Returns", fontsize=10)

    # ── Top-right: metrics bar chart ──────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, n_holds // 2 + 1:])
    metric_names = ["Ann Return", "Sharpe", "Max DD"]
    x = np.arange(len(metric_names))
    bar_w = 0.8 / (n_holds + 1)
    for i, (hk, col) in enumerate(zip(hold_keys, colors)):
        m = hold_variants[hk]["metrics"]
        vals = [m["annualised_return"], m["sharpe"] / 3, -m["max_drawdown"]]
        offset = (i - n_holds / 2) * bar_w
        bars = ax_bar.bar(x + offset, vals, bar_w, color=col, label=f"hold={hk}d", alpha=0.85)
    # benchmark reference line at 0
    ax_bar.axhline(0, color="black", lw=0.5)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metric_names, fontsize=8)
    ax_bar.set_title("Key Metrics (Ann Return | Sharpe/3 | −Max DD)", fontsize=9)
    ax_bar.legend(fontsize=8)

    # ── Bottom row: drawdown per hold variant ─────────────────────────────────
    for j, (hk, col) in enumerate(zip(hold_keys, colors)):
        ax_dd = fig.add_subplot(gs[1, j])
        v = hold_variants[hk]
        dr = pd.Series(v["daily_returns"], index=pd.DatetimeIndex(v["dates"]))
        cum = (1 + dr).cumprod()
        dd  = (cum.cummax() - cum) / cum.cummax()
        ax_dd.fill_between(dd.index, -dd.values, 0, color=col, alpha=_DD_ALPHA)
        ax_dd.plot(dd.index, -dd.values, color=col, lw=1)
        ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax_dd.set_title(f"Drawdown  hold={hk}d  MaxDD={v['metrics']['max_drawdown']:.1%}", fontsize=9)
        ax_dd.set_ylim(bottom=-1)
        # benchmark drawdown overlay
        bm_dd = (bm_cum.cummax() - bm_cum) / bm_cum.cummax()
        ax_dd.plot(bm_dd.index, -bm_dd.values, color=_BM_COLOR, lw=0.9, ls="--", alpha=0.7)

    out_path = out_dir / f"backtest_{data['model_key']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {out_path}")
    return out_path


# ── Main logic ────────────────────────────────────────────────────────────────

def run_backtest(cfg: Config, model_key: str, hold_days_list: list[int]) -> Path:
    specs = build_model_specs(cfg)
    if model_key not in specs:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(specs)}")
    spec = specs[model_key]

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    test_end = str(pd.Timestamp.today().date())
    max_hold = max(hold_days_list)
    min_hold = min(hold_days_list)

    print(f"\n{'='*60}")
    print(f"Model:  {spec.label}")
    print(f"Hold variants: {hold_days_list}  |  pred_len={max_hold}")
    print(f"Period: {cfg.test_start_date} → {test_end}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Close prices for portfolio eval
    close_prices: dict[str, pd.Series] = {}
    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=cfg.test_start_date, end=test_end)
        if len(df) > 0:
            close_prices[sym] = pd.Series(df["close"].values,
                                          index=pd.DatetimeIndex(df["date"]))
    print(f"Loaded close prices: {len(close_prices)} symbols")
    sys.stdout.flush()

    # Benchmark
    bm_df = query_symbol(cfg.db_path, cfg.benchmark_symbol,
                         start=cfg.test_start_date, end=test_end)
    bm_daily = pd.Series(bm_df["close"].values,
                         index=pd.DatetimeIndex(bm_df["date"])).pct_change().dropna()

    # Inference — single pass covering all hold variants
    predictor = load_predictor_from_spec(spec, cfg)
    fine_dates = pd.bdate_range(cfg.test_start_date, test_end)[::min_hold]
    print(f"Inference: {len(fine_dates)} periods × {len(symbols)} symbols")
    sys.stdout.flush()
    raw_preds = compute_raw_signals(predictor, cfg, fine_dates, max_hold, symbols)
    del predictor
    torch.cuda.empty_cache()

    # Build results per hold variant
    hold_variants: dict[str, dict] = {}
    for hd in hold_days_list:
        step = hd // min_hold
        variant_dates = fine_dates[::step]
        holdings = signals_to_holdings(raw_preds, variant_dates, hd, cfg.top_k, cfg.min_signal_threshold)
        _, dr = build_portfolio_returns(close_prices, holdings, variant_dates)
        m = compute_metrics(dr)
        hold_variants[str(hd)] = {
            "dates": [d.strftime("%Y-%m-%d") for d in dr.index],
            "daily_returns": dr.tolist(),
            "metrics": m,
        }
        print(f"  hold={hd}d — Ann:{m['annualised_return']:.2%}  "
              f"Sharpe:{m['sharpe']:.2f}  DD:{m['max_drawdown']:.2%}")
        sys.stdout.flush()

    # Serialize output
    out = {
        "model_key":   model_key,
        "model_label": spec.label,
        "test_start":  cfg.test_start_date,
        "test_end":    test_end,
        "top_k":       cfg.top_k,
        "hold_variants": hold_variants,
        "benchmark": {
            "dates":         [d.strftime("%Y-%m-%d") for d in bm_daily.index],
            "daily_returns": bm_daily.tolist(),
            "metrics":       compute_metrics(bm_daily),
        },
    }

    out_dir = Path(cfg.output_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_returns_{model_key}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")
    sys.stdout.flush()

    plot_backtest_results(out, out_dir)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--model", required=True,
                        choices=["pretrained", "round0", "round1", "round2"],
                        help="Which model weights to load")
    parser.add_argument("--hold_days_list", type=int, nargs="+", default=[5, 10, 15],
                        help="Hold period variants in days (default: 5 10 15)")
    parser.add_argument("--top_k",    type=int, default=None)
    parser.add_argument("--test_start", default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Min predicted return to include a stock (default: config value)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.top_k:      cfg.top_k = args.top_k
    if args.test_start: cfg.test_start_date = args.test_start
    if args.threshold is not None: cfg.min_signal_threshold = args.threshold

    run_backtest(cfg, args.model, args.hold_days_list)


if __name__ == "__main__":
    main()
