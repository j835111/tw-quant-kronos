"""End-to-end stacking backtest runner for Kronos-TW."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from finetune_tw.analog import AnalogEngine
from finetune_tw.backtest import (
    build_model_specs,
    build_portfolio_returns,
    compute_metrics,
    load_predictor_from_spec,
    rank_stocks,
)
from finetune_tw.config import Config
from finetune_tw.db import list_symbols, query_symbol
from finetune_tw.signal import KronosSignalExtractor
from finetune_tw.stacking import FEATURE_COLS, StackingModel, build_feature_row
from finetune_tw.walkforward import WalkForwardFold, oof_folds


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _out_dir(cfg: Config) -> Path:
    out_dir = Path(getattr(cfg, "output_dir", "finetune_tw/outputs")) / getattr(
        cfg, "exp_name", "tw_daily"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _normalize_dates(cfg: Config, date_range: object) -> list[pd.Timestamp]:
    step = max(1, int(getattr(cfg, "hold_days", 5)))

    if date_range is None:
        start = pd.Timestamp(getattr(cfg, "stacking_train_start", "2018-01-01"))
        end = pd.Timestamp(getattr(cfg, "stacking_train_end", "2023-12-31"))
        return list(pd.bdate_range(start, end)[::step])

    if isinstance(date_range, WalkForwardFold):
        start = pd.Timestamp(date_range.val_start)
        end = pd.Timestamp(date_range.val_end)
        if start > end:
            return []
        return list(pd.bdate_range(start, end)[::step])

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1])
        if start > end:
            return []
        return list(pd.bdate_range(start, end)[::step])

    if isinstance(date_range, pd.DatetimeIndex):
        return list(pd.to_datetime(date_range))

    if isinstance(date_range, Iterable) and not isinstance(date_range, (str, bytes)):
        return list(pd.to_datetime(list(date_range)))

    ts = pd.Timestamp(date_range)
    return [ts]


def _load_price_cache(
    cfg: Config,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    lookback_window = int(getattr(cfg, "lookback_window", 90))
    start_str = (start - pd.Timedelta(days=lookback_window * 2)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    db_path = getattr(cfg, "db_path", "finetune_tw/data/tw_stocks.db")

    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        frame = query_symbol(db_path, sym, start=start_str, end=end_str)
        if not frame.empty:
            frames[sym] = frame
    return frames


def _fwd_return_from_frame(
    frame: pd.DataFrame,
    as_of: pd.Timestamp,
    hold_days: int,
) -> float | None:
    sub = frame[pd.to_datetime(frame["date"]) >= as_of].head(hold_days + 1)
    if len(sub) < hold_days + 1:
        return None
    start_close = float(sub["close"].iloc[0])
    end_close = float(sub["close"].iloc[hold_days])
    return end_close / (start_close + 1e-9) - 1.0


def _compute_metrics_compat(daily_returns: pd.Series) -> dict[str, float | int]:
    if daily_returns.empty:
        metrics = {
            "annualised_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
        total_return = 0.0
    else:
        metrics = compute_metrics(daily_returns)
        total_return = float((1.0 + daily_returns).prod() - 1.0)

    metrics["annual_return"] = float(metrics["annualised_return"])
    metrics["sharpe_ratio"] = float(metrics["sharpe"])
    metrics["total_return"] = total_return
    metrics["n_obs"] = int(len(daily_returns))
    return metrics


def _empty_feature_table(include_target: bool) -> pd.DataFrame:
    cols = FEATURE_COLS + (["fwd_return"] if include_target else [])
    empty = pd.DataFrame(columns=cols)
    empty.index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
    return empty


def _save_feature_table(feature_df: pd.DataFrame, path: Path) -> None:
    feature_df.to_parquet(path)


def _fit_analog_engine(
    cfg: Config,
    symbols: list[str],
    cutoff_date: str,
) -> AnalogEngine | None:
    if not getattr(cfg, "analog_enabled", False):
        return None

    engine = AnalogEngine(
        n_neighbors=int(getattr(cfg, "analog_n_neighbors", 20)),
        window=int(getattr(cfg, "analog_window", 20)),
    )
    engine.fit(
        db_path=getattr(cfg, "db_path", "finetune_tw/data/tw_stocks.db"),
        symbols=symbols,
        cutoff_date=cutoff_date,
        pred_len=int(getattr(cfg, "pred_len", 10)),
    )
    return engine


def _build_feature_table(
    cfg: Config,
    extractor: KronosSignalExtractor,
    engine: AnalogEngine | None,
    symbols: list[str],
    dates: list[pd.Timestamp],
    include_target: bool,
    partial_ckpt: Path | None = None,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    if not dates:
        return _empty_feature_table(include_target)

    all_symbols = list(symbols)
    benchmark_symbol = getattr(cfg, "benchmark_symbol", "^TWII")
    if benchmark_symbol not in all_symbols:
        all_symbols.append(benchmark_symbol)

    hold_days = int(getattr(cfg, "hold_days", 5))
    end = max(dates) + pd.offsets.BDay(hold_days + 1)
    frames = _load_price_cache(cfg, all_symbols, min(dates), end)
    bench_frame = frames.get(benchmark_symbol, pd.DataFrame())

    horizon = max(0, min(hold_days, int(getattr(cfg, "pred_len", hold_days))) - 1)

    # Resume from partial checkpoint if available
    rows: list[dict] = []
    done_dates: set[pd.Timestamp] = set()
    if partial_ckpt is not None and partial_ckpt.exists():
        partial_df = pd.read_parquet(partial_ckpt)
        done_dates = set(partial_df.index.get_level_values("date").unique())
        rows = partial_df.reset_index().to_dict("records")
        print(f"[stacking]   partial resume: {len(done_dates)} dates already done", flush=True)

    pending = [d for d in dates if d not in done_dates]
    n_dates = len(dates)

    for di, date in enumerate(pending):
        abs_idx = len(done_dates) + di
        if abs_idx % 20 == 0:
            print(f"[stacking]   date {abs_idx+1}/{n_dates} ({date.date()})", flush=True)
        signals = extractor.extract_date(date, symbols, cfg, horizon=horizon)
        for sym in symbols:
            sym_frame = frames.get(sym, pd.DataFrame())
            if sym_frame.empty:
                continue
            row = build_feature_row(
                sym,
                date,
                signals.get(sym),
                sym_frame,
                bench_frame,
                engine,
                cfg,
            )
            if row is None:
                continue
            if include_target:
                fwd_return = _fwd_return_from_frame(sym_frame, date, hold_days)
                if fwd_return is None:
                    continue
                row["fwd_return"] = float(fwd_return)
            row["date"] = date
            row["symbol"] = sym
            rows.append(row)

        # Save partial checkpoint every N dates
        if partial_ckpt is not None and (di + 1) % checkpoint_every == 0:
            _save_feature_table(
                pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index(),
                partial_ckpt,
            )

    if not rows:
        return _empty_feature_table(include_target)

    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


def _collect_oof_features(
    cfg: Config,
    predictor,
    engine: AnalogEngine | None,
    symbols: list[str],
    date_range,
    partial_ckpt: Path | None = None,
) -> pd.DataFrame:
    inference_top_k = int(getattr(cfg, "mc_inference_top_k",
                                      max(int(getattr(cfg, "top_k", 20)) * 2, 40)))
    extractor = KronosSignalExtractor(
        predictor,
        n_samples=int(getattr(cfg, "mc_sample_count", 20)),
        top_k=inference_top_k,
    )
    dates = _normalize_dates(cfg, date_range)
    return _build_feature_table(cfg, extractor, engine, symbols, dates, include_target=True,
                                partial_ckpt=partial_ckpt)


def _run_test_backtest(
    cfg: Config,
    predictor,
    engine: AnalogEngine | None,
    stacking_model: StackingModel,
    symbols: list[str],
    test_dates,
    out_dir: Path | None = None,
) -> dict:
    dates = _normalize_dates(cfg, test_dates)
    inference_top_k = int(getattr(cfg, "mc_inference_top_k",
                                      max(int(getattr(cfg, "top_k", 20)) * 2, 40)))
    extractor = KronosSignalExtractor(
        predictor,
        n_samples=int(getattr(cfg, "mc_sample_count", 20)),
        top_k=inference_top_k,
    )
    partial_ckpt = (out_dir / "stacking_test_partial.parquet") if out_dir is not None else None
    feature_df = _build_feature_table(
        cfg,
        extractor,
        engine,
        symbols,
        dates,
        include_target=False,
        partial_ckpt=partial_ckpt,
    )
    if partial_ckpt is not None and partial_ckpt.exists():
        partial_ckpt.unlink()

    benchmark_symbol = getattr(cfg, "benchmark_symbol", "^TWII")
    hold_days = int(getattr(cfg, "hold_days", 5))
    if dates:
        close_end = max(dates) + pd.offsets.BDay(hold_days + 1)
        frames = _load_price_cache(cfg, symbols + [benchmark_symbol], min(dates), close_end)
    else:
        frames = {}

    close_prices: dict[str, pd.Series] = {}
    for sym in symbols:
        frame = frames.get(sym, pd.DataFrame())
        if frame.empty:
            continue
        close_prices[sym] = pd.Series(
            frame["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(frame["date"]),
        )

    stacker_holdings: list[set[str]] = []
    kronos_holdings: list[set[str]] = []
    kronos_mc_mean_holdings: list[set[str]] = []
    top_k = int(getattr(cfg, "top_k", 20))

    for date in dates:
        if feature_df.empty:
            stacker_holdings.append(set())
            kronos_holdings.append(set())
            kronos_mc_mean_holdings.append(set())
            continue

        try:
            date_frame = feature_df.xs(date, level="date", drop_level=False)
        except KeyError:
            stacker_holdings.append(set())
            kronos_holdings.append(set())
            kronos_mc_mean_holdings.append(set())
            continue

        scores = stacking_model.predict(date_frame)
        stacker_rank = {
            sym: float(scores.loc[(date, sym)])
            for _, sym in scores.index
            if (date, sym) in scores.index
        }
        kronos_rank = {
            sym: float(date_frame.loc[(date, sym), "kronos_greedy"])
            for _, sym in date_frame.index
            if (date, sym) in date_frame.index
        }
        kronos_mc_mean_rank = {
            sym: float(date_frame.loc[(date, sym), "kronos_mean"])
            for _, sym in date_frame.index
            if (date, sym) in date_frame.index
        }
        stacker_holdings.append(rank_stocks(stacker_rank, top_k))
        kronos_holdings.append(rank_stocks(kronos_rank, top_k))
        kronos_mc_mean_holdings.append(rank_stocks(kronos_mc_mean_rank, top_k))

    _, stacker_daily = build_portfolio_returns(close_prices, stacker_holdings, pd.DatetimeIndex(dates))
    _, kronos_daily = build_portfolio_returns(close_prices, kronos_holdings, pd.DatetimeIndex(dates))
    _, kronos_mc_mean_daily = build_portfolio_returns(close_prices, kronos_mc_mean_holdings, pd.DatetimeIndex(dates))

    bench_frame = frames.get(benchmark_symbol, pd.DataFrame())
    if bench_frame.empty:
        benchmark_daily = pd.Series(dtype=float)
    else:
        benchmark_daily = pd.Series(
            bench_frame["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(bench_frame["date"]),
        ).pct_change().dropna()
        if dates:
            benchmark_daily = benchmark_daily[benchmark_daily.index >= dates[0]]

    return {
        "test_dates": [d.strftime("%Y-%m-%d") for d in dates],
        "stacker": {
            "dates": [d.strftime("%Y-%m-%d") for d in stacker_daily.index],
            "daily_returns": stacker_daily.tolist(),
            "metrics": _compute_metrics_compat(stacker_daily),
        },
        "kronos_only": {
            "dates": [d.strftime("%Y-%m-%d") for d in kronos_daily.index],
            "daily_returns": kronos_daily.tolist(),
            "metrics": _compute_metrics_compat(kronos_daily),
        },
        "kronos_mc_mean": {
            "dates": [d.strftime("%Y-%m-%d") for d in kronos_mc_mean_daily.index],
            "daily_returns": kronos_mc_mean_daily.tolist(),
            "metrics": _compute_metrics_compat(kronos_mc_mean_daily),
        },
        "benchmark": {
            "dates": [d.strftime("%Y-%m-%d") for d in benchmark_daily.index],
            "daily_returns": benchmark_daily.tolist(),
            "metrics": _compute_metrics_compat(benchmark_daily),
        },
    }


def _plot_stacking(data: dict, out_dir: Path, suffix: str = "") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Stacking Backtest — {data['model_label']}  "
        f"({data['test_start']} → {data['test_end']}  top-K={data['top_k']})",
        fontsize=12,
        fontweight="bold",
    )

    colors = {
        "stacker": "#2196F3",
        "kronos_only": "#FF9800",
        "kronos_mc_mean": "#4CAF50",
        "benchmark": "#9E9E9E",
    }
    for name, color in colors.items():
        series = pd.Series(
            data[name]["daily_returns"],
            index=pd.DatetimeIndex(data[name]["dates"]),
        )
        if series.empty:
            continue
        cumulative = (1.0 + series).cumprod()
        metrics = data[name]["metrics"]
        label = (
            f"{name}  Sharpe={metrics['sharpe']:.2f}  "
            f"Ann={metrics['annualised_return']:.1%}"
        )
        axes[0].plot(
            cumulative.index,
            cumulative.values,
            color=color,
            lw=1.8,
            ls="--" if name == "benchmark" else "-",
            label=label,
        )

    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    axes[0].axhline(1.0, color="black", lw=0.6, ls=":")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Cumulative Returns")

    metric_names = ["Ann Return", "Sharpe/3", "-Max DD"]
    x = np.arange(len(metric_names))
    for idx, (name, color) in enumerate(colors.items()):
        metrics = data[name]["metrics"]
        values = [
            metrics["annualised_return"],
            metrics["sharpe"] / 3.0,
            -metrics["max_drawdown"],
        ]
        axes[1].bar(x + (idx - 1) * 0.25, values, 0.25, color=color, label=name, alpha=0.85)
    axes[1].axhline(0.0, color="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_names)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Key Metrics")

    out_path = out_dir / f"backtest_stacking{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_stacking_backtest(cfg: Config, force_retrain: bool = False, suffix: str = "") -> dict:
    if not getattr(cfg, "stacking_enabled", False):
        raise ValueError("cfg.stacking_enabled must be True to run stacking backtest.")
    model_key = getattr(cfg, "model_key", "round0")
    specs = build_model_specs(cfg)
    if model_key not in specs:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(specs)}")
    spec = specs[model_key]

    out_dir = _out_dir(cfg)
    oof_path = out_dir / "stacking_features_oof.parquet"
    model_path = out_dir / "stacking_model.lgb"

    benchmark_symbol = getattr(cfg, "benchmark_symbol", "^TWII")
    symbols = [sym for sym in list_symbols(getattr(cfg, "db_path", "")) if sym != benchmark_symbol]

    if not force_retrain and oof_path.exists() and model_path.exists():
        print(f"[stacking] loading cached OOF features from {oof_path}", flush=True)
        oof_df = pd.read_parquet(oof_path)
        print(f"[stacking] loading cached model from {model_path}", flush=True)
        stacking_model = StackingModel.load(str(model_path))
        predictor = load_predictor_from_spec(spec, cfg)
        print(f"[stacking] cache loaded: {len(oof_df)} OOF rows, skipping retrain", flush=True)
    else:
        predictor = load_predictor_from_spec(spec, cfg)
        combined_oof: list[pd.DataFrame] = []
        folds = oof_folds(
            getattr(cfg, "stacking_train_start", "2018-01-01"),
            getattr(cfg, "stacking_train_end", "2023-12-31"),
            embargo_days=int(getattr(cfg, "wf_embargo_days", 110)),
        )
        print(f"[stacking] {len(symbols)} symbols, {len(folds)} OOF folds", flush=True)
        for fi, fold in enumerate(folds):
            fold_ckpt = out_dir / f"stacking_oof_fold{fi+1}.parquet"
            partial_ckpt = out_dir / f"stacking_oof_fold{fi+1}_partial.parquet"
            if not force_retrain and fold_ckpt.exists():
                print(f"[stacking] fold {fi+1}/{len(folds)}: resuming from checkpoint {fold_ckpt.name}", flush=True)
                fold_df = pd.read_parquet(fold_ckpt)
            else:
                print(f"[stacking] fold {fi+1}/{len(folds)}: {fold.val_start} → {fold.val_end}", flush=True)
                engine = _fit_analog_engine(cfg, symbols, fold.val_start)
                fold_df = _collect_oof_features(
                    cfg, predictor, engine, symbols, fold,
                    partial_ckpt=None if force_retrain else partial_ckpt,
                )
                print(f"[stacking] fold {fi+1} done: {len(fold_df)} rows", flush=True)
                if not fold_df.empty:
                    _save_feature_table(fold_df, fold_ckpt)
                if partial_ckpt.exists():
                    partial_ckpt.unlink()
            if not fold_df.empty:
                combined_oof.append(fold_df)

        if not combined_oof:
            raise ValueError("No OOF stacking features were generated.")

        oof_df = pd.concat(combined_oof).sort_index()
        print(f"[stacking] OOF total: {len(oof_df)} rows → saving parquet", flush=True)
        _save_feature_table(oof_df, oof_path)
        for fi in range(len(folds)):
            fold_ckpt = out_dir / f"stacking_oof_fold{fi+1}.parquet"
            if fold_ckpt.exists():
                fold_ckpt.unlink()

        print("[stacking] fitting LightGBM StackingModel ...", flush=True)
        stacking_model = StackingModel()
        stacking_model.fit(oof_df)
        stacking_model.save(str(model_path))
        print(f"[stacking] model saved → {model_path}", flush=True)

    result: dict = {
        "model_key": f"stacking_{model_key}",
        "model_label": f"Stacking ({getattr(spec, 'label', model_key)})",
        "test_start": getattr(cfg, "test_start_date", "2024-07-01"),
        "test_end": _today().strftime("%Y-%m-%d"),
        "top_k": int(getattr(cfg, "top_k", 20)),
        "hold_days": int(getattr(cfg, "hold_days", 5)),
        "oof_rows": int(len(oof_df)),
        "artifacts": {
            "stacking_model": str(model_path),
            "stacking_features_oof": str(oof_path),
        },
    }

    if getattr(cfg, "train_only", False):
        json_path = out_dir / "backtest_stacking.json"
        json_path.write_text(json.dumps(result, indent=2))
        result["artifacts"]["backtest_json"] = str(json_path)
        return result

    test_dates = pd.bdate_range(getattr(cfg, "test_start_date", "2024-07-01"), _today())[
        :: max(1, int(getattr(cfg, "hold_days", 5)))
    ]
    print(f"[stacking] test backtest: {len(test_dates)} dates from {getattr(cfg, 'test_start_date', '2024-07-01')}", flush=True)
    test_engine = _fit_analog_engine(cfg, symbols, getattr(cfg, "test_start_date", "2024-07-01"))
    test_result = _run_test_backtest(cfg, predictor, test_engine, stacking_model, symbols, test_dates,
                                     out_dir=out_dir)
    print("[stacking] test backtest done", flush=True)
    result.update(test_result)

    json_path = out_dir / f"backtest_stacking{suffix}.json"
    json_path.write_text(json.dumps(result, indent=2))
    png_path = _plot_stacking(result, out_dir, suffix=suffix)
    result["artifacts"]["backtest_json"] = str(json_path)
    result["artifacts"]["backtest_png"] = str(png_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument(
        "--model",
        default="round0",
        choices=["pretrained", "round0", "round1", "round2"],
    )
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--no-analog", action="store_true")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Ignore cached OOF parquet/model and recompute from scratch")
    parser.add_argument("--mc", type=int, default=None,
                        help="Override mc_sample_count from config")
    parser.add_argument("--inference-top-k", type=int, default=None,
                        help="Override inference top_k for KronosSignalExtractor (default: max(cfg.top_k*2, 40))")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to output filenames (e.g. _mc3)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.model_key = args.model
    cfg.train_only = args.train_only
    if args.no_analog:
        cfg.analog_enabled = False
    if args.mc is not None:
        cfg.mc_sample_count = args.mc
    if args.inference_top_k is not None:
        cfg.mc_inference_top_k = args.inference_top_k

    run_stacking_backtest(cfg, force_retrain=args.force_retrain, suffix=args.suffix)


if __name__ == "__main__":
    main()
