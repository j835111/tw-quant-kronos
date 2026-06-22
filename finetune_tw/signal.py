"""Kronos Monte Carlo signal extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from model import KronosPredictor

from finetune_tw.config import Config
from finetune_tw.db import query_symbol

_BATCH_SIZE = 32
_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


@dataclass
class KronosSignal:
    mean_return: float
    q10: float
    q50: float
    q90: float
    dispersion: float
    dir_prob: float


class KronosSignalExtractor:
    def __init__(
        self,
        predictor: KronosPredictor,
        n_samples: int = 20,
        top_k: int = 40,
        temperature: float = 1.0,
    ) -> None:
        self.predictor = predictor
        self.n_samples = n_samples
        self.top_k = top_k
        self.temperature = temperature

    def _load_context(
        self,
        sym: str,
        as_of: pd.Timestamp,
        cfg: Config,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
        start = (as_of - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
        end = as_of.strftime("%Y-%m-%d")
        df = query_symbol(cfg.db_path, sym, start=start, end=end)
        if len(df) < cfg.lookback_window:
            return None

        ctx = df.iloc[-cfg.lookback_window:]
        ctx_df = ctx[_PRICE_COLUMNS].reset_index(drop=True)
        if ctx_df.isnull().any().any():
            return None

        x_ts = pd.to_datetime(ctx["date"]).reset_index(drop=True)
        y_ts = pd.Series(pd.date_range(as_of, periods=cfg.pred_len, freq="B"))
        return ctx_df, x_ts, y_ts

    def extract_date(
        self,
        date: pd.Timestamp,
        symbols: list[str],
        cfg: Config,
        horizon: int = 4,
    ) -> dict[str, KronosSignal]:
        contexts: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series, float]] = {}
        for sym in symbols:
            loaded = self._load_context(sym, date, cfg)
            if loaded is None:
                continue
            ctx_df, x_ts, y_ts = loaded
            contexts[sym] = (ctx_df, x_ts, y_ts, float(ctx_df["close"].iloc[-1]))

        if not contexts:
            return {}

        sym_list = list(contexts)
        sample_returns: dict[str, list[float]] = {sym: [] for sym in sym_list}
        df_list = [contexts[sym][0] for sym in sym_list]
        x_ts_list = [contexts[sym][1] for sym in sym_list]
        y_ts_list = [contexts[sym][2] for sym in sym_list]
        last_closes = {sym: contexts[sym][3] for sym in sym_list}

        with torch.no_grad():
            for _ in range(self.n_samples):
                for start in range(0, len(sym_list), _BATCH_SIZE):
                    stop = start + _BATCH_SIZE
                    batch_syms = sym_list[start:stop]
                    preds = self.predictor.predict_batch(
                        df_list=df_list[start:stop],
                        x_timestamp_list=x_ts_list[start:stop],
                        y_timestamp_list=y_ts_list[start:stop],
                        pred_len=cfg.pred_len,
                        T=self.temperature,
                        top_k=self.top_k,
                        top_p=1.0,
                        sample_count=1,
                        verbose=False,
                    )
                    for sym, pred in zip(batch_syms, preds):
                        if pred is None or len(pred) <= horizon:
                            continue
                        ret = float(pred["close"].iloc[horizon]) / last_closes[sym] - 1.0
                        sample_returns[sym].append(ret)

        results: dict[str, KronosSignal] = {}
        for sym, returns in sample_returns.items():
            arr = np.asarray(returns, dtype=float)
            if len(arr) == 0:
                continue
            results[sym] = KronosSignal(
                mean_return=float(arr.mean()),
                q10=float(np.percentile(arr, 10)),
                q50=float(np.percentile(arr, 50)),
                q90=float(np.percentile(arr, 90)),
                dispersion=float(arr.std()),
                dir_prob=float((arr > 0).mean()),
            )
        return results

    def extract_date_range(
        self,
        dates: list[pd.Timestamp],
        symbols: list[str],
        cfg: Config,
        horizon: int = 4,
    ) -> pd.DataFrame:
        rows: list[dict[str, float | pd.Timestamp | str]] = []
        for date in dates:
            signals = self.extract_date(date, symbols, cfg, horizon=horizon)
            for sym, sig in signals.items():
                rows.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "kronos_mean": sig.mean_return,
                        "kronos_q10": sig.q10,
                        "kronos_q50": sig.q50,
                        "kronos_q90": sig.q90,
                        "kronos_disp": sig.dispersion,
                        "kronos_dir_prob": sig.dir_prob,
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index(["date", "symbol"])
