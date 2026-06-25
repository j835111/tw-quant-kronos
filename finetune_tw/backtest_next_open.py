from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch

from finetune_tw.backtest import (
    build_model_specs,
    compute_metrics,
    compute_raw_signals,
    load_predictor_from_spec,
    rank_stocks,
    signals_to_holdings,
)
from finetune_tw.config import Config
from finetune_tw.db import list_symbols, query_symbol


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _load_trading_calendar(cfg: Config, end: str) -> pd.DatetimeIndex:
    bm_df = query_symbol(
        cfg.db_path,
        cfg.benchmark_symbol,
        start=cfg.test_start_date,
        end=end,
    )
    if bm_df.empty:
        raise ValueError(
            f"No benchmark rows found for {cfg.benchmark_symbol} between "
            f"{cfg.test_start_date} and {end}."
        )
    return pd.DatetimeIndex(pd.to_datetime(bm_df["date"]))


def _build_signal_and_execution_dates(
    trading_dates: pd.DatetimeIndex,
    hold_days: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if hold_days <= 0:
        raise ValueError(f"hold_days must be positive, got {hold_days}")

    signal_dates = trading_dates[::hold_days]
    kept_signal_dates: list[pd.Timestamp] = []
    execution_dates: list[pd.Timestamp] = []

    for signal_date in signal_dates:
        idx = trading_dates.get_loc(signal_date)
        if idx + 1 >= len(trading_dates):
            continue
        kept_signal_dates.append(signal_date)
        execution_dates.append(trading_dates[idx + 1])

    return pd.DatetimeIndex(kept_signal_dates), pd.DatetimeIndex(execution_dates)


def _load_price_frames(
    cfg: Config,
    symbols: list[str],
    end: str,
) -> dict[str, pd.DataFrame]:
    price_frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=cfg.test_start_date, end=end)
        if df.empty:
            continue
        frame = df.loc[:, ["date", "open", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        price_frames[sym] = frame
    return price_frames


def _mean_symbol_return(
    price_frames: dict[str, pd.DataFrame],
    symbols: set[str],
    numerator_date: pd.Timestamp,
    numerator_col: str,
    denominator_date: pd.Timestamp,
    denominator_col: str,
) -> float | None:
    values: list[float] = []
    for sym in symbols:
        frame = price_frames.get(sym)
        if frame is None:
            continue
        if numerator_date not in frame.index or denominator_date not in frame.index:
            continue
        den = float(frame.loc[denominator_date, denominator_col])
        if den == 0.0:
            continue
        num = float(frame.loc[numerator_date, numerator_col])
        values.append(num / den - 1.0)
    if not values:
        return None
    return float(np.mean(values))


def build_next_open_portfolio_returns(
    price_frames: dict[str, pd.DataFrame],
    holdings_sequence: list[set[str]],
    execution_dates: pd.DatetimeIndex,
    trading_dates: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    if len(holdings_sequence) != len(execution_dates):
        raise ValueError(
            "holdings_sequence and execution_dates must have the same length."
        )
    if len(execution_dates) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    daily_values: list[float] = []
    daily_index: list[pd.Timestamp] = []
    period_values: list[float] = []
    period_index: list[pd.Timestamp] = []

    first_intraday = _mean_symbol_return(
        price_frames,
        holdings_sequence[0],
        execution_dates[0],
        "close",
        execution_dates[0],
        "open",
    )
    if first_intraday is not None:
        daily_values.append(first_intraday)
        daily_index.append(execution_dates[0])

    for i in range(len(execution_dates) - 1):
        current_exec = execution_dates[i]
        next_exec = execution_dates[i + 1]
        current_holdings = holdings_sequence[i]
        next_holdings = holdings_sequence[i + 1]

        interior_dates = trading_dates[
            (trading_dates > current_exec) & (trading_dates < next_exec)
        ]
        prev_date = current_exec
        for date in interior_dates:
            close_to_close = _mean_symbol_return(
                price_frames,
                current_holdings,
                date,
                "close",
                prev_date,
                "close",
            )
            if close_to_close is not None:
                daily_values.append(close_to_close)
                daily_index.append(date)
            prev_date = date

        gap = _mean_symbol_return(
            price_frames,
            current_holdings,
            next_exec,
            "open",
            prev_date,
            "close",
        )
        intraday = _mean_symbol_return(
            price_frames,
            next_holdings,
            next_exec,
            "close",
            next_exec,
            "open",
        )
        if gap is not None and intraday is not None:
            daily_values.append((1.0 + gap) * (1.0 + intraday) - 1.0)
            daily_index.append(next_exec)

        period_return = _mean_symbol_return(
            price_frames,
            current_holdings,
            next_exec,
            "open",
            current_exec,
            "open",
        )
        period_values.append(period_return if period_return is not None else 0.0)
        period_index.append(current_exec)

    return (
        pd.Series(period_values, index=pd.DatetimeIndex(period_index), dtype=float),
        pd.Series(daily_values, index=pd.DatetimeIndex(daily_index), dtype=float),
    )


_HOLD_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]
_BM_COLOR = "#9E9E9E"
_DD_ALPHA = 0.18


def plot_backtest_next_open_results(data: dict, out_dir: Path) -> Path:
    hold_variants = data["hold_variants"]
    bm = data["benchmark"]
    model_label = data["model_label"]
    hold_keys = sorted(hold_variants, key=int)

    bm_dates = pd.DatetimeIndex(bm["dates"])
    bm_cum = (1 + pd.Series(bm["daily_returns"], index=bm_dates)).cumprod()

    n_holds = len(hold_keys)
    colors = (_HOLD_COLORS * 4)[:n_holds]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Backtest — {model_label}  ({data['test_start']} → {data['test_end']}  top-K={data['top_k']})",
        fontsize=13,
        fontweight="bold",
    )

    ncols = max(n_holds, 2)
    cum_cols = 1 if ncols == 2 else n_holds // 2 + 1
    gs = fig.add_gridspec(2, ncols, hspace=0.38, wspace=0.32)

    ax_cum = fig.add_subplot(gs[0, :cum_cols])
    ax_cum.plot(
        bm_cum.index,
        bm_cum.values,
        color=_BM_COLOR,
        lw=1.5,
        linestyle="--",
        label=f"^TWII  Sharpe={bm['metrics']['sharpe']:.2f}",
    )
    for hk, col in zip(hold_keys, colors):
        variant = hold_variants[hk]
        dr = pd.Series(variant["daily_returns"], index=pd.DatetimeIndex(variant["dates"]))
        cum = (1 + dr).cumprod()
        metrics = variant["metrics"]
        ax_cum.plot(
            cum.index,
            cum.values,
            color=col,
            lw=1.8,
            label=f"hold={hk}d  Sharpe={metrics['sharpe']:.2f}  Ann={metrics['annualised_return']:.1%}",
        )
    ax_cum.set_ylabel("Cumulative Return")
    ax_cum.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax_cum.axhline(1, color="black", lw=0.6, ls=":")
    ax_cum.legend(fontsize=7.5, loc="upper left")
    ax_cum.set_title("Cumulative Returns", fontsize=10)

    ax_bar = fig.add_subplot(gs[0, cum_cols:])
    metric_names = ["Ann Return", "Sharpe", "Max DD"]
    x = np.arange(len(metric_names))
    bar_w = 0.8 / (n_holds + 1)
    for i, (hk, col) in enumerate(zip(hold_keys, colors)):
        metrics = hold_variants[hk]["metrics"]
        vals = [metrics["annualised_return"], metrics["sharpe"] / 3, -metrics["max_drawdown"]]
        offset = (i - n_holds / 2) * bar_w
        ax_bar.bar(x + offset, vals, bar_w, color=col, label=f"hold={hk}d", alpha=0.85)
    ax_bar.axhline(0, color="black", lw=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, fontsize=8)
    ax_bar.set_title("Key Metrics (Ann Return | Sharpe/3 | −Max DD)", fontsize=9)
    ax_bar.legend(fontsize=8)

    for j, (hk, col) in enumerate(zip(hold_keys, colors)):
        ax_dd = fig.add_subplot(gs[1, j])
        variant = hold_variants[hk]
        dr = pd.Series(variant["daily_returns"], index=pd.DatetimeIndex(variant["dates"]))
        cum = (1 + dr).cumprod()
        dd = (cum.cummax() - cum) / cum.cummax()
        ax_dd.fill_between(dd.index, -dd.values, 0, color=col, alpha=_DD_ALPHA)
        ax_dd.plot(dd.index, -dd.values, color=col, lw=1)
        ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax_dd.set_title(
            f"Drawdown  hold={hk}d  MaxDD={variant['metrics']['max_drawdown']:.1%}",
            fontsize=9,
        )
        ax_dd.set_ylim(bottom=-1)
        bm_dd = (bm_cum.cummax() - bm_cum) / bm_cum.cummax()
        ax_dd.plot(bm_dd.index, -bm_dd.values, color=_BM_COLOR, lw=0.9, ls="--", alpha=0.7)

    out_path = out_dir / f"backtest_{data['model_key']}_next_open.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {out_path}")
    return out_path


def run_backtest_next_open(cfg: Config, model_key: str, hold_days_list: list[int]) -> Path:
    specs = build_model_specs(cfg)
    if model_key not in specs:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(specs)}")
    spec = specs[model_key]

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    test_end = str(_today().date())
    max_hold = max(hold_days_list)

    print(f"\n{'=' * 60}")
    print(f"Model:  {spec.label}")
    print(f"Hold variants: {hold_days_list}  |  pred_len={max_hold}")
    print(f"Period: {cfg.test_start_date} → {test_end}")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    trading_dates = _load_trading_calendar(cfg, test_end)
    variant_schedules = {
        hd: _build_signal_and_execution_dates(trading_dates, hold_days=hd)
        for hd in hold_days_list
    }
    all_signal_dates = sorted(
        {signal_date for signal_dates, _ in variant_schedules.values() for signal_date in signal_dates}
    )
    signal_dates = pd.DatetimeIndex(all_signal_dates)
    price_frames = _load_price_frames(cfg, symbols, test_end)
    print(f"Loaded open/close prices: {len(price_frames)} symbols")
    sys.stdout.flush()

    bm_df = query_symbol(
        cfg.db_path,
        cfg.benchmark_symbol,
        start=cfg.test_start_date,
        end=test_end,
    )
    bm_frame = bm_df.loc[:, ["date", "close"]].copy()
    bm_frame["date"] = pd.to_datetime(bm_frame["date"])
    bm_frame = bm_frame.set_index("date").sort_index().reindex(trading_dates)
    bm_daily = bm_frame["close"].pct_change().dropna()

    predictor = load_predictor_from_spec(spec, cfg)
    print(f"Inference: {len(signal_dates)} periods × {len(symbols)} symbols")
    sys.stdout.flush()
    raw_preds = compute_raw_signals(predictor, cfg, signal_dates, max_hold, symbols)
    del predictor
    torch.cuda.empty_cache()

    hold_variants: dict[str, dict] = {}
    for hd in hold_days_list:
        variant_signal_dates, variant_execution_dates = variant_schedules[hd]
        holdings = signals_to_holdings(
            raw_preds,
            variant_signal_dates,
            hd,
            cfg.top_k,
            cfg.min_signal_threshold,
        )
        _, daily_returns = build_next_open_portfolio_returns(
            price_frames=price_frames,
            holdings_sequence=holdings,
            execution_dates=variant_execution_dates,
            trading_dates=trading_dates,
        )
        if daily_returns.empty:
            raise ValueError(
                f"No realized daily returns for hold_days={hd}. "
                "Check trading calendar coverage and execution dates."
            )
        metrics = compute_metrics(daily_returns)
        hold_variants[str(hd)] = {
            "dates": [d.strftime("%Y-%m-%d") for d in daily_returns.index],
            "daily_returns": daily_returns.tolist(),
            "metrics": metrics,
        }
        print(
            f"  hold={hd}d — Ann:{metrics['annualised_return']:.2%}  "
            f"Sharpe:{metrics['sharpe']:.2f}  DD:{metrics['max_drawdown']:.2%}"
        )
        sys.stdout.flush()

    out = {
        "model_key": model_key,
        "model_label": spec.label,
        "test_start": cfg.test_start_date,
        "test_end": test_end,
        "top_k": cfg.top_k,
        "hold_variants": hold_variants,
        "benchmark": {
            "dates": [d.strftime("%Y-%m-%d") for d in bm_daily.index],
            "daily_returns": bm_daily.tolist(),
            "metrics": compute_metrics(bm_daily),
        },
    }

    out_dir = Path(cfg.output_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_returns_{model_key}_next_open.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")
    sys.stdout.flush()

    plot_backtest_next_open_results(out, out_dir)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument(
        "--model",
        required=True,
        choices=["pretrained", "round0", "round1", "round2"],
        help="Which model weights to load",
    )
    parser.add_argument(
        "--hold_days_list",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="Hold period variants in days (default: 5 10 15)",
    )
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--test_start", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Min predicted return to include a stock (default: config value)",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.top_k:
        cfg.top_k = args.top_k
    if args.test_start:
        cfg.test_start_date = args.test_start
    if args.threshold is not None:
        cfg.min_signal_threshold = args.threshold

    run_backtest_next_open(cfg, args.model, args.hold_days_list)


if __name__ == "__main__":
    main()
