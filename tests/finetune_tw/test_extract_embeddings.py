import numpy as np
import pandas as pd
import pytest
import torch

from model.kronos import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.extract_embeddings import _select_symbols, compute_technical_features, extract_embeddings_batch


def _make_tiny_predictor() -> KronosPredictor:
    torch.manual_seed(0)
    tokenizer = KronosTokenizer(
        d_in=6, d_model=8, n_heads=2, ff_dim=16,
        n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=2, s2_bits=2, beta=1e-2, gamma0=1.0, gamma=1.0, zeta=1.0,
        group_size=1,
    )
    model = Kronos(
        s1_bits=2, s2_bits=2, n_layers=1, d_model=8, n_heads=2, ff_dim=16,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=False,
    )
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=64)
    return predictor


def _make_tiny_predictor_multi_layer(n_layers: int = 3) -> KronosPredictor:
    torch.manual_seed(0)
    tokenizer = KronosTokenizer(
        d_in=6, d_model=8, n_heads=2, ff_dim=16,
        n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=2, s2_bits=2, beta=1e-2, gamma0=1.0, gamma=1.0, zeta=1.0,
        group_size=1,
    )
    model = Kronos(
        s1_bits=2, s2_bits=2, n_layers=n_layers, d_model=8, n_heads=2, ff_dim=16,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=False,
    )
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device="cpu", max_context=64)


def _make_df(offset: float, n: int = 20) -> pd.DataFrame:
    idx = np.arange(n, dtype=np.float32)
    return pd.DataFrame({
        "open": 10.0 + offset + idx * 0.1,
        "high": 10.5 + offset + idx * 0.1,
        "low": 9.5 + offset + idx * 0.1,
        "close": 10.2 + offset + idx * 0.1,
        "volume": 100.0 + idx,
        "amount": 1000.0 + idx * 10,
    })


def test_extract_embeddings_batch_shape():
    predictor = _make_tiny_predictor()
    df_list = [_make_df(0.0), _make_df(5.0), _make_df(-3.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    embeddings = extract_embeddings_batch(predictor, df_list, [x_ts, x_ts, x_ts])
    assert embeddings.shape == (3, 8)  # (batch, d_model)
    assert np.isfinite(embeddings).all()


def test_extract_embeddings_batch_is_deterministic_in_eval_mode():
    predictor = _make_tiny_predictor()
    df_list = [_make_df(0.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    first = extract_embeddings_batch(predictor, df_list, [x_ts])
    second = extract_embeddings_batch(predictor, df_list, [x_ts])
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_extract_embeddings_batch_default_layer_indices_matches_old_shape():
    predictor = _make_tiny_predictor_multi_layer(n_layers=3)
    df_list = [_make_df(0.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    embeddings = extract_embeddings_batch(predictor, df_list, [x_ts])
    assert embeddings.shape == (1, 8)  # unchanged: (batch, d_model), backward compatible


def test_extract_embeddings_batch_multi_layer_concatenates_selected_layers():
    predictor = _make_tiny_predictor_multi_layer(n_layers=3)
    df_list = [_make_df(0.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))

    single_layer = extract_embeddings_batch(predictor, df_list, [x_ts], layer_indices=[0])
    two_layers = extract_embeddings_batch(predictor, df_list, [x_ts], layer_indices=[0, 2])

    assert single_layer.shape == (1, 8)
    assert two_layers.shape == (1, 16)  # 2 layers * d_model=8
    np.testing.assert_allclose(two_layers[:, :8], single_layer, rtol=1e-5, atol=1e-6)


def _make_df_with_shape(pattern: str, n: int = 20) -> pd.DataFrame:
    """Two genuinely different normalized shapes (not just a level/offset shift, which z-score
    normalization removes entirely — a pure additive offset would make two inputs indistinguishable
    after normalization by construction, so that's not a valid way to test this function)."""
    idx = np.arange(n, dtype=np.float32)
    if pattern == "uptrend":
        base = idx * 0.5
    else:  # "oscillating"
        base = np.sin(idx * 0.8) * 3.0
    return pd.DataFrame({
        "open": 10.0 + base,
        "high": 10.5 + base,
        "low": 9.5 + base,
        "close": 10.2 + base,
        "volume": 100.0 + idx,
        "amount": 1000.0 + idx * 10,
    })


def test_extract_embeddings_batch_distinguishes_different_inputs():
    predictor = _make_tiny_predictor()
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    df_list = [_make_df_with_shape("uptrend"), _make_df_with_shape("oscillating")]
    embeddings = extract_embeddings_batch(predictor, df_list, [x_ts, x_ts])
    assert not np.allclose(embeddings[0], embeddings[1])


def test_select_symbols_none_returns_full_universe():
    symbols = ["1101", "2330", "2454"]
    assert _select_symbols(symbols, None) == symbols


def test_select_symbols_truncates_to_first_n_sorted():
    symbols = ["1101", "1102", "2330", "2454"]
    assert _select_symbols(symbols, 2) == ["1101", "1102"]


def test_compute_technical_features_matches_hand_calculation():
    n = 70
    idx = np.arange(n, dtype=np.float64)
    close = 100.0 + idx
    volume = 200.0 + idx
    volumes = volume.copy()
    volumes[-1] = 800.0
    df = pd.DataFrame({
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": volumes,
        "amount": np.full(n, 20000.0),
    })

    feats = compute_technical_features(df)

    last_close = close[-1]
    returns = close[1:] / close[:-1] - 1.0
    expected_ma5 = close[-5:].mean()
    expected_ma20 = close[-20:].mean()
    expected_momentum_10 = last_close / close[-11] - 1.0
    expected_vol20 = volumes[-20:].mean()

    assert feats["feat_ma5_dist"] == pytest.approx(last_close / expected_ma5 - 1.0)
    assert feats["feat_ma20_dist"] == pytest.approx(last_close / expected_ma20 - 1.0)
    assert feats["feat_ma60_dist"] == pytest.approx(last_close / close[-60:].mean() - 1.0)
    assert feats["feat_momentum_3"] == pytest.approx(last_close / close[-4] - 1.0)
    assert feats["feat_momentum_5"] == pytest.approx(last_close / close[-6] - 1.0)
    assert feats["feat_momentum_10"] == pytest.approx(expected_momentum_10)
    assert feats["feat_momentum_20"] == pytest.approx(last_close / close[-21] - 1.0)
    assert feats["feat_momentum_60"] == pytest.approx(last_close / close[-61] - 1.0)
    assert feats["feat_vol_10"] == pytest.approx(returns[-10:].std())
    assert feats["feat_vol_30"] == pytest.approx(returns[-30:].std())
    assert feats["feat_volume_ratio"] == pytest.approx(800.0 / expected_vol20)
    assert feats["feat_volume_trend"] == pytest.approx(volumes[-5:].mean() / expected_vol20)
    assert feats["feat_hl_spread_5"] == pytest.approx((((close + 0.5) - (close - 0.5)) / close)[-5:].mean())


def test_compute_technical_features_handles_short_history():
    n = 3
    df = pd.DataFrame({
        "open": [10.0, 11.0, 12.0], "high": [10.5, 11.5, 12.5],
        "low": [9.5, 10.5, 11.5], "close": [10.0, 11.0, 12.0],
        "volume": [100.0, 100.0, 100.0], "amount": [1000.0, 1000.0, 1000.0],
    })
    feats = compute_technical_features(df)  # must not raise IndexError with < 5/20/11 rows
    assert set(feats) == {
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
    }
    assert all(np.isfinite(v) for v in feats.values())
