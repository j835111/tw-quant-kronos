from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
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
    build_portfolio_returns,
    compute_metrics,
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


def compute_raw_signals_open(
    predictor,
    cfg: Config,
    rebal_dates: pd.DatetimeIndex,
    pred_len: int,
    symbols: list[str],
    attach_pred_frame: bool = False,
) -> dict[str, dict[str, pd.Series]]:
    """Like compute_raw_signals but ranks by predicted open[T+h+1]/open[T+1]-1.

    pred_len must be max_hold + 1 so iloc[h] is available for each hold variant h.
    The returned Series has length pred_len-1, where iloc[h-1] = open[T+h+1]/open[T+1]-1,
    aligning exactly with next-open execution for hold=h days.
    """
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
                    df_list=batch_dfs[b : b + BATCH_SIZE],
                    x_timestamp_list=batch_xts[b : b + BATCH_SIZE],
                    y_timestamp_list=batch_yts[b : b + BATCH_SIZE],
                    pred_len=pred_len,
                    T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False,
                )
                for sym, pred in zip(batch_syms[b : b + BATCH_SIZE], preds):
                    if pred is not None and len(pred) >= pred_len:
                        pred_opens = pred["open"].reset_index(drop=True)
                        # iloc[h-1] = open[T+h+1] / open[T+1] - 1
                        returns = (
                            pred_opens.iloc[1:].reset_index(drop=True) / pred_opens.iloc[0] - 1
                        )
                        if attach_pred_frame:
                            returns.attrs["pred_frame"] = pred.loc[
                                :, ["high", "low", "close"]
                            ].reset_index(drop=True)
                        date_preds[sym] = returns

        raw_preds[rebal_str] = date_preds
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(rebal_dates)}] {rebal_str}: {len(date_preds)} signals")
            sys.stdout.flush()

    return raw_preds


compute_raw_signals = compute_raw_signals_open


def _predicted_atr_from_series(
    pred_returns: pd.Series | None,
    hold_days: int,
    min_atr: float,
) -> float | None:
    if pred_returns is None:
        return None
    pred_frame = pred_returns.attrs.get("pred_frame")
    if not isinstance(pred_frame, pd.DataFrame) or pred_frame.empty:
        return None
    if not {"high", "low", "close"}.issubset(pred_frame.columns):
        return None

    h = min(max(hold_days - 1, 0), len(pred_frame) - 1)
    high = float(pred_frame["high"].iloc[h])
    low = float(pred_frame["low"].iloc[h])
    close = float(pred_frame["close"].iloc[h])
    if not np.isfinite(high) or not np.isfinite(low) or not np.isfinite(close):
        return None
    if close <= 0:
        return min_atr
    return float(max((high - low) / close, min_atr))


def compute_atr_weights(
    raw_preds: dict[str, pd.Series] | None,
    hold_days: int,
    selected_syms: list[str],
    min_atr: float = 0.003,
) -> dict[str, float]:
    if not selected_syms:
        return {}

    date_preds = raw_preds or {}
    weights: dict[str, float] = {}
    for sym in selected_syms:
        pred_atr = _predicted_atr_from_series(date_preds.get(sym), hold_days, min_atr)
        weights[sym] = 1.0 / pred_atr if pred_atr is not None else 1.0

    total = float(sum(weights.values()))
    if total <= 0:
        equal_weight = 1.0 / len(selected_syms)
        return {sym: equal_weight for sym in selected_syms}
    return {sym: weight / total for sym, weight in weights.items()}


def _mean_symbol_return(
    price_frames: dict[str, pd.DataFrame],
    symbols: set[str],
    numerator_date: pd.Timestamp,
    numerator_col: str,
    denominator_date: pd.Timestamp,
    denominator_col: str,
    weights: dict[str, float] | None = None,
) -> float | None:
    values: list[float] = []
    weighted_values: list[float] = []
    valid_weights: list[float] = []
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
        ret = num / den - 1.0
        if weights is None:
            values.append(ret)
            continue
        weight = float(weights.get(sym, 0.0))
        if weight <= 0:
            continue
        weighted_values.append(ret * weight)
        valid_weights.append(weight)
    if weights is None:
        if not values:
            return None
        return float(np.mean(values))
    total_weight = float(sum(valid_weights))
    if total_weight <= 0:
        return None
    return float(sum(weighted_values) / total_weight)


def _resolve_period_weights(
    weights: dict | None,
    period_date: pd.Timestamp,
) -> dict[str, float] | None:
    if not weights:
        return None
    sample = next(iter(weights.values()), None)
    if isinstance(sample, Mapping):
        date_key = pd.Timestamp(period_date)
        period_weights = weights.get(date_key)
        if period_weights is None:
            period_weights = weights.get(date_key.strftime("%Y-%m-%d"))
        return dict(period_weights) if period_weights else None
    return dict(weights)


def build_next_open_portfolio_returns(
    price_frames: dict[str, pd.DataFrame],
    holdings_sequence: list[set[str]],
    execution_dates: pd.DatetimeIndex,
    trading_dates: pd.DatetimeIndex,
    weights: dict | None = None,
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

    first_weights = _resolve_period_weights(weights, execution_dates[0])
    first_intraday = _mean_symbol_return(
        price_frames,
        holdings_sequence[0],
        execution_dates[0],
        "close",
        execution_dates[0],
        "open",
        weights=first_weights,
    )
    if first_intraday is not None:
        daily_values.append(first_intraday)
        daily_index.append(execution_dates[0])

    for i in range(len(execution_dates) - 1):
        current_exec = execution_dates[i]
        next_exec = execution_dates[i + 1]
        current_holdings = holdings_sequence[i]
        next_holdings = holdings_sequence[i + 1]
        current_weights = _resolve_period_weights(weights, current_exec)
        next_weights = _resolve_period_weights(weights, next_exec)

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
                weights=current_weights,
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
            weights=current_weights,
        )
        intraday = _mean_symbol_return(
            price_frames,
            next_holdings,
            next_exec,
            "close",
            next_exec,
            "open",
            weights=next_weights,
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
            weights=current_weights,
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


def run_backtest_next_open(
    cfg: Config,
    model_key: str,
    hold_days_list: list[int],
    use_atr_weights: bool = False,
) -> Path:
    specs = build_model_specs(cfg)
    if model_key not in specs:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(specs)}")
    spec = specs[model_key]

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    test_end = str(_today().date())
    max_hold = max(hold_days_list)

    pred_len = max_hold + 1  # +1 so open[T+h+1] is available for hold=h

    print(f"\n{'=' * 60}")
    print(f"Model:  {spec.label}")
    print(
        f"Hold variants: {hold_days_list}  |  pred_len={pred_len}  "
        f"(open-to-open signal, weights={'ATR' if use_atr_weights else 'equal'})"
    )
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
    raw_preds = compute_raw_signals(
        predictor,
        cfg,
        signal_dates,
        pred_len,
        symbols,
        attach_pred_frame=use_atr_weights,
    )
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
        period_weights: dict[str, dict[str, float]] | None = None
        if use_atr_weights:
            period_weights = {}
            atr_samples: list[float] = []
            for signal_date, execution_date, selected_syms in zip(
                variant_signal_dates,
                variant_execution_dates,
                holdings,
            ):
                selected_list = sorted(selected_syms)
                date_preds = raw_preds.get(signal_date.strftime("%Y-%m-%d"), {})
                period_weights[execution_date.strftime("%Y-%m-%d")] = compute_atr_weights(
                    date_preds,
                    hd,
                    selected_list,
                )
                for sym in selected_list:
                    pred_atr = _predicted_atr_from_series(date_preds.get(sym), hd, 0.003)
                    if pred_atr is not None:
                        atr_samples.append(pred_atr)
            if atr_samples:
                print(
                    f"  hold={hd}d ATR stats — min:{min(atr_samples):.4f}  "
                    f"mean:{float(np.mean(atr_samples)):.4f}  max:{max(atr_samples):.4f}"
                )
                sys.stdout.flush()
        build_kwargs = {
            "price_frames": price_frames,
            "holdings_sequence": holdings,
            "execution_dates": variant_execution_dates,
            "trading_dates": trading_dates,
        }
        if period_weights is not None:
            build_kwargs["weights"] = period_weights
        _, daily_returns = build_next_open_portfolio_returns(**build_kwargs)
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
    parser.add_argument(
        "--atr-weights",
        action="store_true",
        help="Use normalized inverse-predicted-ATR weights instead of equal weights",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.top_k:
        cfg.top_k = args.top_k
    if args.test_start:
        cfg.test_start_date = args.test_start
    if args.threshold is not None:
        cfg.min_signal_threshold = args.threshold

    run_backtest_next_open(
        cfg,
        args.model,
        args.hold_days_list,
        use_atr_weights=args.atr_weights,
    )


if __name__ == "__main__":
    main()
