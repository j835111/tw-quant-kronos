import torch
import pandas as pd

from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices
from finetune_tw.train_predictor import _build_ctx_for_date, _resolve_amp


def test_resolve_amp_bf16():
    enabled, dtype = _resolve_amp("bf16")
    assert enabled is True
    assert dtype == torch.bfloat16


def test_resolve_amp_none():
    enabled, dtype = _resolve_amp("none")
    assert enabled is False
    assert dtype is None


def test_resolve_amp_fp16():
    enabled, dtype = _resolve_amp("fp16")
    assert enabled is True
    assert dtype == torch.float16


def test_resolve_amp_unknown_falls_back_to_disabled():
    enabled, dtype = _resolve_amp("tf32")
    assert enabled is False
    assert dtype is None


def test_build_ctx_for_date_shapes(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    dates = pd.bdate_range("2023-06-01", periods=200)
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": [100.0 + i * 0.1 for i in range(200)],
            "volume": 1000.0,
            "amount": 1e5,
        }
    )
    upsert_prices(db, "9999", df)
    cfg = Config(db_path=db, lookback_window=90, pred_len=10)

    built = _build_ctx_for_date(cfg, "9999", pd.Timestamp("2024-01-15"))

    assert built is not None
    ctx_df, x_ts, y_ts, last_date, ctx_close = built
    assert len(ctx_df) == 90
    assert list(ctx_df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert len(x_ts) == 90
    assert len(y_ts) == cfg.pred_len
    assert last_date == x_ts.iloc[-1]
    assert ctx_close == ctx_df["close"].iloc[-1]


def test_build_ctx_for_date_insufficient_returns_none(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    dates = pd.bdate_range("2023-12-01", periods=10)
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
            "amount": 1e5,
        }
    )
    upsert_prices(db, "9999", df)
    cfg = Config(db_path=db, lookback_window=90, pred_len=10)

    assert _build_ctx_for_date(cfg, "9999", pd.Timestamp("2024-01-15")) is None
