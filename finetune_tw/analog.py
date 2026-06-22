"""Analog Engine: point-in-time k-NN retrieval of historically similar windows.

Invariant: ALL windows in the retrieval index have their forward outcomes already
realized at query time. Enforced by: cutoff_date = as_of − pred_len*2 calendar days.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from finetune_tw.db import query_symbol


@dataclass
class AnalogFeatures:
    fwd_q25: float
    fwd_q50: float
    fwd_q75: float
    up_prob: float       # P(forward return > 0) among analogs
    max_gain: float
    max_loss: float
    dispersion: float    # std of analog forward returns = confidence signal
    n_analogs: int


class AnalogEngine:
    """k-NN retrieval engine for 'look-alike' historical windows."""

    def __init__(self, n_neighbors: int = 20, window: int = 20) -> None:
        self.n_neighbors = n_neighbors
        self.window = window
        self._keys: np.ndarray = np.empty((0, 0))
        self._fwd_returns: np.ndarray = np.empty(0)

    def _featurize(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Convert a price/volume window into a shape-normalized retrieval key.
        
        Returns a vector of length window+1:
          - (window-1) normalized log returns
          - vol_z: volume z-score of last bar vs window
          - range_feat: (max-min)/mean of the window
        """
        if len(close) < 2:
            return np.zeros(self.window + 1)
        log_rets = np.diff(np.log(close + 1e-9))
        if log_rets.std() > 1e-9:
            log_rets = (log_rets - log_rets.mean()) / log_rets.std()
        # Pad/truncate to exactly window-1 elements
        n = self.window - 1
        if len(log_rets) >= n:
            log_rets = log_rets[-n:]
        else:
            log_rets = np.pad(log_rets, (n - len(log_rets), 0))
        vol_z = (volume[-1] - volume.mean()) / (volume.std() + 1e-9)
        range_feat = (close.max() - close.min()) / (close.mean() + 1e-9)
        return np.concatenate([log_rets, [vol_z, range_feat]])

    def fit(
        self,
        db_path: str,
        symbols: list[str],
        cutoff_date: str,
        pred_len: int = 10,
        start_date: str = "2015-01-01",
    ) -> "AnalogEngine":
        """Build retrieval index. All windows end strictly before cutoff_date."""
        strict_cutoff = (
            pd.Timestamp(cutoff_date) - pd.Timedelta(days=pred_len * 2)
        ).strftime("%Y-%m-%d")

        keys, fwd_returns = [], []
        for sym in symbols:
            df = query_symbol(db_path, sym, start=start_date, end=strict_cutoff)
            if len(df) < self.window + pred_len + 1:
                continue
            close = df["close"].values.astype(float)
            volume = df["volume"].values.astype(float)
            for i in range(len(df) - self.window - pred_len):
                win_close = close[i:i + self.window]
                win_vol = volume[i:i + self.window]
                fwd_last = close[i + self.window + pred_len - 1]
                fwd_ret = fwd_last / (win_close[-1] + 1e-9) - 1.0
                keys.append(self._featurize(win_close, win_vol))
                fwd_returns.append(fwd_ret)

        if keys:
            self._keys = np.array(keys, dtype=float)
            self._fwd_returns = np.array(fwd_returns, dtype=float)
        return self

    def query(
        self,
        recent_close: np.ndarray,
        recent_volume: np.ndarray,
    ) -> "AnalogFeatures | None":
        """Return statistics of forward returns from k nearest analog windows."""
        if self._keys.shape[0] == 0:
            return None
        key = self._featurize(recent_close, recent_volume)
        k = min(self.n_neighbors, len(self._keys))
        dists = np.linalg.norm(self._keys - key, axis=1)
        idx = np.argpartition(dists, k - 1)[:k]
        fwd = self._fwd_returns[idx]
        return AnalogFeatures(
            fwd_q25=float(np.percentile(fwd, 25)),
            fwd_q50=float(np.percentile(fwd, 50)),
            fwd_q75=float(np.percentile(fwd, 75)),
            up_prob=float((fwd > 0).mean()),
            max_gain=float(fwd.max()),
            max_loss=float(fwd.min()),
            dispersion=float(fwd.std()),
            n_analogs=k,
        )
