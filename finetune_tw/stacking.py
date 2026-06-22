"""LightGBM cross-sectional stacking model for Kronos-TW signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from finetune_tw.features import (
    build_market_relative_features,
    build_tech_features,
)

if TYPE_CHECKING:
    from finetune_tw.analog import AnalogEngine, AnalogFeatures
    from finetune_tw.signal import KronosSignal


_KRONOS_COLS = [
    "kronos_mean",
    "kronos_q10",
    "kronos_q50",
    "kronos_q90",
    "kronos_disp",
    "kronos_dir_prob",
]
_TECH_COLS = [
    "ma20_gap",
    "ma60_gap",
    "rsi_14",
    "bb_pct",
    "mom_10d",
    "mom_20d",
    "vol_20d",
]
_MARKET_COLS = ["alpha_20d", "alpha_60d", "rel_vol"]
_ANALOG_COLS = [
    "analog_q25",
    "analog_q50",
    "analog_q75",
    "analog_up_prob",
    "analog_max_gain",
    "analog_max_loss",
    "analog_disp",
]

FEATURE_COLS: list[str] = _KRONOS_COLS + _TECH_COLS + _MARKET_COLS + _ANALOG_COLS

_LGBM_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}


def build_feature_row(
    sym: str,
    as_of: pd.Timestamp,
    kronos_signal: "KronosSignal | None",
    sym_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    analog_engine: "AnalogEngine | None",
    cfg: Any,
) -> dict[str, float] | None:
    """Build one ordered feature row for a (date, symbol) pair.

    Returns None if tech or market-relative features are unavailable.
    """
    del sym

    lookback = getattr(cfg, "lookback_window", 90)
    tech = build_tech_features(sym_df, as_of, lookback=lookback)
    if tech is None:
        return None

    market = build_market_relative_features(sym_df, bench_df, as_of, lookback=lookback)
    if market is None:
        return None

    row: dict[str, float] = {}

    if kronos_signal is None:
        row.update({col: 0.0 for col in _KRONOS_COLS})
    else:
        row.update(
            {
                "kronos_mean": float(kronos_signal.mean_return),
                "kronos_q10": float(kronos_signal.q10),
                "kronos_q50": float(kronos_signal.q50),
                "kronos_q90": float(kronos_signal.q90),
                "kronos_disp": float(kronos_signal.dispersion),
                "kronos_dir_prob": float(kronos_signal.dir_prob),
            }
        )

    row.update({col: float(tech[col]) for col in _TECH_COLS})
    row.update({col: float(market[col]) for col in _MARKET_COLS})

    analog_features: "AnalogFeatures | None" = None
    if analog_engine is not None:
        cutoff_mask = pd.to_datetime(sym_df["date"]) <= as_of
        window = getattr(analog_engine, "window", 0)
        analog_slice = sym_df.loc[cutoff_mask].tail(window)
        if window > 0 and len(analog_slice) == window:
            analog_features = analog_engine.query(
                analog_slice["close"].to_numpy(dtype=float, copy=True),
                analog_slice["volume"].to_numpy(dtype=float, copy=True),
            )

    if analog_features is None:
        row.update({col: 0.0 for col in _ANALOG_COLS})
    else:
        row.update(
            {
                "analog_q25": float(analog_features.fwd_q25),
                "analog_q50": float(analog_features.fwd_q50),
                "analog_q75": float(analog_features.fwd_q75),
                "analog_up_prob": float(analog_features.up_prob),
                "analog_max_gain": float(analog_features.max_gain),
                "analog_max_loss": float(analog_features.max_loss),
                "analog_disp": float(analog_features.dispersion),
            }
        )

    return {col: row[col] for col in FEATURE_COLS}


class StackingModel:
    """Cross-sectional LightGBM lambdarank stacking model."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_rounds: int = 200,
    ) -> None:
        self.params = dict(_LGBM_PARAMS if params is None else params)
        self.num_rounds = num_rounds
        self._booster: lgb.Booster | None = None

    def fit(self, feature_df: pd.DataFrame) -> "StackingModel":
        """Train the lambdarank model on cross-sectional data.

        feature_df must have MultiIndex (date, symbol) and a 'fwd_return' column.
        """
        df = feature_df.copy()
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df = df.dropna(subset=["fwd_return"])
        df = df.sort_index(level="date")

        def _quintile_label(s: pd.Series) -> pd.Series:
            ranks = s.rank(method="first")
            return ((ranks - 1) / len(s) * 5).clip(0, 4).astype(int)

        df["_label"] = df.groupby(level="date")["fwd_return"].transform(_quintile_label)

        X = df[FEATURE_COLS].to_numpy(dtype=float)
        y = df["_label"].to_numpy(dtype=float)
        groups = df.groupby(level="date").size().values.tolist()

        dtrain = lgb.Dataset(X, label=y, group=groups)
        self._booster = lgb.train(self.params, dtrain, num_boost_round=self.num_rounds)
        return self

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        if self._booster is None:
            raise RuntimeError("StackingModel must be fitted before calling predict().")

        df = feature_df.copy()
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df = df.sort_index(level="date")
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        scores = self._booster.predict(X)
        return pd.Series(np.asarray(scores, dtype=float), index=df.index, name="score")

    def save(self, path: str) -> None:
        if self._booster is None:
            raise RuntimeError("Cannot save an unfitted model.")
        self._booster.save_model(path)

    @classmethod
    def load(cls, path: str) -> "StackingModel":
        obj = cls.__new__(cls)
        obj.params = dict(_LGBM_PARAMS)
        obj.num_rounds = 200
        obj._booster = lgb.Booster(model_file=path)
        return obj
