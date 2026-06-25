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
