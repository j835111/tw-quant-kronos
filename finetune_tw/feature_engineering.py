from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


TECH_FEATURE_COLUMNS = [
    "feat_ma5_dist",
    "feat_ma20_dist",
    "feat_ma60_dist",
    "feat_momentum_3",
    "feat_momentum_5",
    "feat_momentum_10",
    "feat_momentum_20",
    "feat_momentum_60",
    "feat_vol_10",
    "feat_vol_30",
    "feat_volume_ratio",
    "feat_volume_trend",
    "feat_hl_spread_5",
]


def technical_feature_columns(columns: Iterable[str]) -> list[str]:
    """Canonical feat_* column order, with cs-rank features immediately after their base column."""
    cols = list(columns)
    col_set = set(cols)
    ordered: list[str] = []
    for base in TECH_FEATURE_COLUMNS:
        if base in col_set:
            ordered.append(base)
        rank_col = f"{base}_cs_rank"
        if rank_col in col_set:
            ordered.append(rank_col)
    seen = set(ordered)
    extras = sorted(c for c in cols if c.startswith("feat_") and c not in seen)
    return ordered + extras


def compute_technical_features(df: pd.DataFrame) -> dict[str, float]:
    """Technical features for one lookback window ending at the current rebalance date."""
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    last_close = float(close[-1])

    ma5 = float(close[-5:].mean()) if len(close) >= 5 else float(close.mean())
    ma20 = float(close[-20:].mean()) if len(close) >= 20 else float(close.mean())
    ma60 = float(close[-60:].mean()) if len(close) >= 60 else float(close.mean())

    def momentum(n: int) -> float:
        idx = -(n + 1)
        if len(close) <= n or close[idx] == 0:
            return 0.0
        return float(last_close / close[idx] - 1.0)

    returns = close[1:] / close[:-1] - 1.0
    vol_10 = float(returns[-10:].std()) if len(returns) >= 10 else 0.0
    vol_30 = float(returns[-30:].std()) if len(returns) >= 30 else 0.0

    recent_vol_mean = float(volume[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
    volume_trend = (
        float(volume[-5:].mean() / recent_vol_mean)
        if len(volume) >= 20 and recent_vol_mean != 0
        else 1.0
    )

    high_low = np.divide(
        high - low,
        close,
        out=np.zeros_like(close, dtype=np.float64),
        where=close != 0,
    )

    return {
        "feat_ma5_dist": float(last_close / ma5 - 1.0) if ma5 != 0 else 0.0,
        "feat_ma20_dist": float(last_close / ma20 - 1.0) if ma20 != 0 else 0.0,
        "feat_ma60_dist": float(last_close / ma60 - 1.0) if ma60 != 0 else 0.0,
        "feat_momentum_3": momentum(3),
        "feat_momentum_5": momentum(5),
        "feat_momentum_10": momentum(10),
        "feat_momentum_20": momentum(20),
        "feat_momentum_60": momentum(60),
        "feat_vol_10": vol_10,
        "feat_vol_30": vol_30,
        "feat_volume_ratio": float(volume[-1] / recent_vol_mean) if recent_vol_mean != 0 else 1.0,
        "feat_volume_trend": volume_trend,
        "feat_hl_spread_5": float(high_low[-5:].mean()),
    }


def compute_technical_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Vectorized technical features for a symbol/date history table."""
    if history.empty:
        return pd.DataFrame(columns=["symbol", "date", *TECH_FEATURE_COLUMNS])

    df = history.copy().sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    counts = df.groupby("symbol", sort=False).cumcount() + 1

    grouped_close = df.groupby("symbol", sort=False)["close"]
    grouped_volume = df.groupby("symbol", sort=False)["volume"]

    ma5 = grouped_close.transform(lambda s: s.rolling(5, min_periods=1).mean())
    ma20 = grouped_close.transform(lambda s: s.rolling(20, min_periods=1).mean())
    ma60 = grouped_close.transform(lambda s: s.rolling(60, min_periods=1).mean())

    def _momentum(n: int) -> pd.Series:
        shifted = grouped_close.shift(n)
        return np.where(
            shifted.notna() & (shifted != 0),
            df["close"] / shifted - 1.0,
            0.0,
        )

    returns = grouped_close.pct_change()
    vol_10 = returns.groupby(df["symbol"], sort=False).transform(
        lambda s: s.rolling(10, min_periods=10).std(ddof=0)
    ).fillna(0.0)
    vol_30 = returns.groupby(df["symbol"], sort=False).transform(
        lambda s: s.rolling(30, min_periods=30).std(ddof=0)
    ).fillna(0.0)

    vol20_mean = grouped_volume.transform(lambda s: s.rolling(20, min_periods=1).mean())
    vol5_mean = grouped_volume.transform(lambda s: s.rolling(5, min_periods=1).mean())

    high_low = np.divide(
        df["high"] - df["low"],
        df["close"],
        out=np.zeros(len(df), dtype=np.float64),
        where=df["close"].to_numpy(dtype=np.float64) != 0,
    )
    hl_spread_5 = pd.Series(high_low, index=df.index).groupby(df["symbol"], sort=False).transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    feat_df = pd.DataFrame({
        "symbol": df["symbol"].values,
        "date": df["date"].astype(str).str.slice(0, 10).values,
        "feat_ma5_dist": np.where(ma5 != 0, df["close"] / ma5 - 1.0, 0.0),
        "feat_ma20_dist": np.where(ma20 != 0, df["close"] / ma20 - 1.0, 0.0),
        "feat_ma60_dist": np.where(ma60 != 0, df["close"] / ma60 - 1.0, 0.0),
        "feat_momentum_3": _momentum(3),
        "feat_momentum_5": _momentum(5),
        "feat_momentum_10": _momentum(10),
        "feat_momentum_20": _momentum(20),
        "feat_momentum_60": _momentum(60),
        "feat_vol_10": vol_10.to_numpy(dtype=np.float64),
        "feat_vol_30": vol_30.to_numpy(dtype=np.float64),
        "feat_volume_ratio": np.where(vol20_mean != 0, df["volume"] / vol20_mean, 1.0),
        "feat_volume_trend": np.where(
            (counts >= 20) & (vol20_mean != 0),
            vol5_mean / vol20_mean,
            1.0,
        ),
        "feat_hl_spread_5": hl_spread_5.to_numpy(dtype=np.float64),
    })
    return feat_df


def add_cross_sectional_rank_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Append per-date percentile ranks for the supplied feat_* columns."""
    if df.empty:
        return df.copy()

    out = df.copy()
    if feature_cols is None:
        feature_cols = [c for c in technical_feature_columns(out.columns) if not c.endswith("_cs_rank")]
    for col in feature_cols:
        if col not in out.columns or col.endswith("_cs_rank"):
            continue
        out[f"{col}_cs_rank"] = out.groupby(date_col)[col].rank(pct=True)
    return out


def compute_ranked_feature_block(
    history: pd.DataFrame,
    block_keys: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    assume_sorted: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    """Compute technical features only for requested block keys from a bounded history window."""
    if block_keys.empty:
        columns = ["date", "symbol", *(feature_cols or TECH_FEATURE_COLUMNS)]
        columns += [f"{col}_cs_rank" for col in (feature_cols or TECH_FEATURE_COLUMNS)]
        return pd.DataFrame(columns=columns)

    cols = feature_cols or TECH_FEATURE_COLUMNS
    keys = block_keys.loc[:, ["date", "symbol"]].drop_duplicates().copy()
    keys["date"] = keys["date"].astype(str).str.slice(0, 10)
    history_view = history
    if not assume_sorted:
        history_view = history.sort_values(["symbol", "date"], kind="stable")

    records: list[dict[str, float | str]] = []
    target_dates_by_symbol = {
        symbol: symbol_keys["date"].drop_duplicates().tolist()
        for symbol, symbol_keys in keys.groupby("symbol", sort=False)
    }
    for symbol, symbol_history in history_view.groupby("symbol", sort=False):
        target_dates = target_dates_by_symbol.get(symbol)
        if not target_dates:
            continue
        symbol_dates = symbol_history["date"]
        if pd.api.types.is_datetime64_any_dtype(symbol_dates):
            date_values = symbol_dates.dt.strftime("%Y-%m-%d").to_numpy()
        else:
            date_values = symbol_dates.astype(str).str.slice(0, 10).to_numpy()
        price_history = symbol_history.loc[:, ["open", "high", "low", "close", "volume", "amount"]]
        for target_date in target_dates:
            end = date_values.searchsorted(target_date, side="right")
            if end == 0 or date_values[end - 1] != target_date:
                continue
            feature_values = compute_technical_features(
                price_history.iloc[:end]
            )
            records.append({
                "date": target_date,
                "symbol": symbol,
                **{col: feature_values[col] for col in cols},
            })

    feature_df = pd.DataFrame.from_records(records, columns=["date", "symbol", *cols])
    selected = keys.merge(feature_df, on=["date", "symbol"], how="left", validate="1:1", sort=False)
    missing = selected[cols].isna().any(axis=1)
    if missing.any():
        if strict:
            sample = selected.loc[missing, ["date", "symbol"]].head(5).to_dict("records")
            raise ValueError(f"missing recomputed technical features for some parquet rows: {sample}")
        selected = selected.loc[~missing].reset_index(drop=True)
        if selected.empty:
            columns = ["date", "symbol", *cols, *[f"{col}_cs_rank" for col in cols]]
            return pd.DataFrame(columns=columns)
    return add_cross_sectional_rank_features(selected, feature_cols=cols)
