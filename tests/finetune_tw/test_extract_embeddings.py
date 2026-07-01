import numpy as np
import pandas as pd
import torch

from model.kronos import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.extract_embeddings import extract_embeddings_batch


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
