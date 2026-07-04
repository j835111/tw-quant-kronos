import numpy as np
import pandas as pd
import pytest

from finetune_tw.backtest_xgb_embedding import compute_xgb_signals, run_backtest_xgb_embedding
from finetune_tw.config import Config
from finetune_tw.backtest import rank_stocks, signals_to_holdings
from finetune_tw.backtest_xgb_embedding import xgb_signals_to_raw_preds


def test_xgb_signals_to_raw_preds_plugs_into_signals_to_holdings():
    xgb_preds_by_date = {
        "2024-01-02": {"A": 0.05, "B": 0.02, "C": -0.01, "D": 0.10},
    }
    raw_preds = xgb_signals_to_raw_preds(xgb_preds_by_date, hold_days=5)

    # Every symbol's series must be long enough for iloc[hold_days - 1] and equal the raw score.
    for sym, series in raw_preds["2024-01-02"].items():
        assert len(series) == 5
        assert series.iloc[4] == xgb_preds_by_date["2024-01-02"][sym]

    holdings = signals_to_holdings(raw_preds, pd.DatetimeIndex(["2024-01-02"]), hold_days=5, top_k=2)
    assert holdings == [{"D", "A"}]  # top 2 by score: D=0.10, A=0.05


def test_xgb_signals_to_raw_preds_is_hold_days_invariant():
    xgb_preds_by_date = {"2024-01-02": {"A": 0.05, "B": 0.02}}
    for hold_days in (3, 5, 10):
        raw_preds = xgb_signals_to_raw_preds(xgb_preds_by_date, hold_days=hold_days)
        assert raw_preds["2024-01-02"]["A"].iloc[hold_days - 1] == 0.05


def _make_history_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    return pd.DataFrame(
        {"open": [10.0, 10.5], "high": [10.2, 10.7], "low": [9.8, 10.3], "close": [10.1, 10.6]},
        index=idx,
    )


def _make_context_df(base: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [base, base + 1.0],
            "high": [base + 0.2, base + 1.2],
            "low": [base - 0.2, base + 0.8],
            "close": [base + 0.1, base + 1.1],
            "volume": [100.0, 110.0],
            "amount": [1000.0, 1100.0],
        }
    )


class _RecordingBooster:
    def __init__(self, pred_col: int = 0):
        self.pred_col = pred_col
        self.calls: list[np.ndarray] = []

    def predict(self, matrix):
        arr = np.asarray(matrix, dtype=np.float32)
        self.calls.append(arr.copy())
        return arr[:, self.pred_col]


@pytest.mark.parametrize("feature_columns", [None, ["emb_0", "feat_ma5_dist"]])
def test_compute_xgb_signals_skips_cs_rank_and_predicts_per_batch_when_not_needed(monkeypatch, feature_columns):
    cfg = Config(lookback_window=2)
    rebal_dates = pd.DatetimeIndex(["2024-01-02"])
    history = {"AAA": _make_history_frame(), "BBB": _make_history_frame()}
    context_dfs = [_make_context_df(1.0), _make_context_df(2.0)]
    timestamps = [pd.Series(rebal_dates), pd.Series(rebal_dates)]

    monkeypatch.setattr("finetune_tw.backtest_xgb_embedding.BATCH_SIZE", 1)
    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.load_symbol_history_frames",
        lambda *args, **kwargs: history,
    )
    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.build_rebalance_inputs",
        lambda *args, **kwargs: (["AAA", "BBB"], context_dfs, timestamps, None),
    )
    state = {"extract_calls": 0}

    def _extract_embeddings_batch(predictor, df_list, x_ts_list):
        state["extract_calls"] += 1
        if state["extract_calls"] == 2:
            assert len(booster.calls) == 1, "fast path must predict the first sub-batch before embedding the second"
        return np.array(
            [[1.0, 10.0]] if df_list[0]["open"].iloc[0] == 1.0 else [[2.0, 20.0]],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.extract_embeddings_batch",
        _extract_embeddings_batch,
    )
    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.compute_technical_features",
        lambda ctx_df: {"feat_ma5_dist": float(ctx_df["open"].iloc[0]) / 10.0},
    )
    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.add_cross_sectional_rank_features",
        lambda df: (_ for _ in ()).throw(AssertionError("cs-rank path should be bypassed")),
    )
    monkeypatch.setattr("finetune_tw.backtest_xgb_embedding.xgb.DMatrix", lambda features: features)

    booster = _RecordingBooster(pred_col=0)
    signals = compute_xgb_signals(object(), booster, cfg, rebal_dates, ["AAA", "BBB"], feature_columns=feature_columns)

    assert signals == {"2024-01-02": {"AAA": 1.0, "BBB": 2.0}}
    expected_width = 3 if feature_columns is None else len(feature_columns)
    assert [call.shape for call in booster.calls] == [(1, expected_width), (1, expected_width)]


def test_run_backtest_xgb_embedding_requires_summary_feature_columns(tmp_path):
    cfg = Config()
    xgb_model_path = tmp_path / "model.json"
    xgb_model_path.write_text("{}")

    with pytest.raises(ValueError, match=r"\.summary\.json.*feature_columns"):
        run_backtest_xgb_embedding(cfg, "pretrained", str(xgb_model_path), [5], 10)


def test_compute_xgb_signals_uses_cs_rank_materialization_when_rank_features_requested(monkeypatch):
    cfg = Config(lookback_window=2)
    rebal_dates = pd.DatetimeIndex(["2024-01-02"])
    history = {"AAA": _make_history_frame(), "BBB": _make_history_frame()}
    context_dfs = [_make_context_df(1.0), _make_context_df(2.0)]
    timestamps = [pd.Series(rebal_dates), pd.Series(rebal_dates)]

    monkeypatch.setattr("finetune_tw.backtest_xgb_embedding.BATCH_SIZE", 1)
    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.load_symbol_history_frames",
        lambda *args, **kwargs: history,
    )
    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.build_rebalance_inputs",
        lambda *args, **kwargs: (["AAA", "BBB"], context_dfs, timestamps, None),
    )
    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.extract_embeddings_batch",
        lambda predictor, df_list, x_ts_list: np.array(
            [[1.0, 10.0]] if df_list[0]["open"].iloc[0] == 1.0 else [[2.0, 20.0]],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.compute_technical_features",
        lambda ctx_df: {"feat_ma5_dist": float(ctx_df["open"].iloc[0]) / 10.0},
    )
    monkeypatch.setattr(
        "finetune_tw.backtest_xgb_embedding.add_cross_sectional_rank_features",
        lambda df: df.assign(feat_ma5_dist_cs_rank=[0.9, 0.1]),
    )
    monkeypatch.setattr("finetune_tw.backtest_xgb_embedding.xgb.DMatrix", lambda features: features)

    booster = _RecordingBooster(pred_col=1)
    signals = compute_xgb_signals(
        object(),
        booster,
        cfg,
        rebal_dates,
        ["AAA", "BBB"],
        feature_columns=["emb_0", "feat_ma5_dist_cs_rank"],
    )

    assert signals["2024-01-02"]["AAA"] == pytest.approx(0.9)
    assert signals["2024-01-02"]["BBB"] == pytest.approx(0.1)
    assert [call.shape for call in booster.calls] == [(2, 2)]
