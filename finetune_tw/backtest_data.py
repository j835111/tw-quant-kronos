from __future__ import annotations

import pandas as pd

from finetune_tw.db import query_symbols_window


_OHLCVA_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
_OHLCVA_COLUMN_SET = set(_OHLCVA_COLUMNS)


def _validate_field(field: str) -> None:
    if field not in _OHLCVA_COLUMN_SET:
        allowed = ", ".join(_OHLCVA_COLUMNS)
        raise ValueError(f"Expected field to be one of: {allowed}")


def _validate_fields(fields: list[str]) -> None:
    invalid = [field for field in fields if field not in _OHLCVA_COLUMN_SET]
    if invalid:
        allowed = ", ".join(_OHLCVA_COLUMNS)
        invalid_text = ", ".join(invalid)
        raise ValueError(
            f"Expected fields to be drawn from: {allowed}. Got invalid fields: {invalid_text}"
        )


def load_symbol_history_frames(
    db_path: str,
    symbols: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Load OHLCVA history once and return per-symbol frames in input order."""
    history = query_symbols_window(db_path, symbols, start=start, end=end)
    if history.empty:
        return {}

    grouped_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in history.groupby("symbol", sort=False):
        symbol_frame = frame.loc[:, ["date", *_OHLCVA_COLUMNS]].copy()
        symbol_frame["date"] = pd.to_datetime(symbol_frame["date"])
        grouped_frames[symbol] = symbol_frame.set_index("date").sort_index()

    return {
        symbol: grouped_frames[symbol]
        for symbol in symbols
        if symbol in grouped_frames
    }


def load_price_field_series(
    db_path: str,
    symbols: list[str],
    start: str,
    end: str,
    field: str,
) -> dict[str, pd.Series]:
    """Return one OHLCVA field per symbol as a Series."""
    _validate_field(field)
    frames = load_symbol_history_frames(db_path, symbols, start=start, end=end)
    return {symbol: frame[field].copy() for symbol, frame in frames.items()}


def load_price_frame_fields(
    db_path: str,
    symbols: list[str],
    start: str,
    end: str,
    fields: list[str],
) -> dict[str, pd.DataFrame]:
    """Return selected OHLCVA fields per symbol as DataFrames."""
    _validate_fields(fields)
    frames = load_symbol_history_frames(db_path, symbols, start=start, end=end)
    return {symbol: frame.loc[:, fields].copy() for symbol, frame in frames.items()}


def build_rebalance_inputs(
    history_frames: dict[str, pd.DataFrame],
    symbols: list[str],
    rebal_date: pd.Timestamp,
    lookback_window: int,
    pred_len: int,
) -> tuple[list[str], list[pd.DataFrame], list[pd.Series], list[pd.Series]]:
    """Build one rebalance batch from preloaded history, skipping short or NaN contexts."""
    batch_syms: list[str] = []
    batch_dfs: list[pd.DataFrame] = []
    batch_xts: list[pd.Series] = []
    batch_yts: list[pd.Series] = []
    future_dates = pd.date_range(rebal_date, periods=pred_len, freq="B")

    for symbol in symbols:
        frame = history_frames.get(symbol)
        if frame is None or frame.empty:
            continue

        history = frame.loc[frame.index <= rebal_date, _OHLCVA_COLUMNS]
        if len(history) < lookback_window:
            continue

        context = history.iloc[-lookback_window:].copy()
        if context.isnull().any().any():
            continue

        batch_syms.append(symbol)
        batch_dfs.append(context.reset_index(drop=True))
        batch_xts.append(pd.Series(context.index, dtype="datetime64[ns]"))
        batch_yts.append(pd.Series(future_dates, dtype="datetime64[ns]"))

    return batch_syms, batch_dfs, batch_xts, batch_yts
