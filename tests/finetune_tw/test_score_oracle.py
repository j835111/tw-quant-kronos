import numpy as np
import pytest
import torch
from unittest.mock import MagicMock


def _make_fake_tokenizer(s1_bits=4, s2_bits=4):
    tok = MagicMock()
    tok.s1_bits = s1_bits
    v_s1 = 2 ** s1_bits
    v_s2 = 2 ** s2_bits

    def fake_encode(x, half=False):
        batch_size, time_steps, _ = x.shape
        s1 = torch.randint(0, v_s1, (batch_size, time_steps))
        s2 = torch.randint(0, v_s2, (batch_size, time_steps))
        return s1, s2

    tok.encode = fake_encode
    return tok, v_s1


def _make_pre_sliced_samples(n_samples=200, lookback=10, pred_len=6, vocab_size=16):
    total_steps = lookback + pred_len + 1
    samples = []
    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        opens = 100 * np.cumprod(1 + rng.normal(0, 0.01, total_steps))
        samples.append(
            {
                "s1_ids": torch.randint(0, vocab_size, (lookback,), dtype=torch.long),
                "open_prices": torch.tensor(opens, dtype=torch.float32),
            }
        )
    return samples


def _make_dict_samples():
    return [
        {
            "s1_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "open_prices": torch.tensor([100.0, 101.0, 102.0, 105.0, 110.0, 115.0], dtype=torch.float32),
        },
        {
            "s1_ids": torch.tensor([4, 5, 2], dtype=torch.long),
            "open_prices": torch.tensor([90.0, 92.0, 94.0, 96.0, 99.0, 102.0], dtype=torch.float32),
        },
        {
            "s1_ids": torch.tensor([7, 8, 9], dtype=torch.long),
            "open_prices": torch.tensor([80.0, 81.0, 82.0, 83.0, 84.0, 90.0], dtype=torch.float32),
        },
    ]


def test_build_s1_oracle_from_samples_shape_and_type():
    from finetune_tw.score_oracle import build_s1_oracle_from_samples

    dataset = _make_pre_sliced_samples(n_samples=300, lookback=10, pred_len=6, vocab_size=16)

    oracle = build_s1_oracle_from_samples(dataset, horizon=5, min_count=5, vocab_size=16)

    assert oracle.shape == (16,)
    assert oracle.dtype == torch.float32
    assert torch.isfinite(oracle).all()


def test_build_s1_oracle_from_raw_dict_samples_uses_mean_return():
    from finetune_tw.score_oracle import build_s1_oracle_from_samples

    samples = _make_dict_samples()

    oracle = build_s1_oracle_from_samples(samples, horizon=2, min_count=2, vocab_size=16)

    expected = (((115.0 / 105.0) - 1.0) + ((102.0 / 96.0) - 1.0)) / 2.0
    assert oracle[2].item() == pytest.approx(expected, abs=1e-7)
    assert oracle[9].item() == 0.0


def test_build_s1_oracle_signature_accepts_raw_samples_in_db_path_slot():
    from finetune_tw.score_oracle import build_s1_oracle

    tok, _ = _make_fake_tokenizer(s1_bits=4)
    samples = _make_dict_samples()

    oracle = build_s1_oracle(
        tok,
        samples,
        start="2020-01-01",
        end="2020-12-31",
        lookback=3,
        predict_window=3,
        horizon=2,
        clip=5.0,
        seed=42,
        min_count=2,
    )

    assert oracle.shape == (16,)
    assert oracle[2].item() != 0.0


def test_oracle_pred_score_differentiable():
    from finetune_tw.score_oracle import oracle_pred_score

    oracle = torch.randn(16)
    s1_logits = torch.randn(8, 16, requires_grad=True)

    scores = oracle_pred_score(s1_logits, oracle)

    assert scores.shape == (8,)
    scores.sum().backward()
    assert s1_logits.grad is not None
    assert not torch.all(s1_logits.grad == 0)


def test_oracle_tokens_with_few_samples_get_zero():
    from finetune_tw.score_oracle import build_s1_oracle_from_samples

    dataset = _make_pre_sliced_samples(n_samples=1, lookback=10, pred_len=6, vocab_size=16)

    oracle = build_s1_oracle_from_samples(dataset, horizon=5, min_count=20, vocab_size=16)

    assert oracle.abs().sum().item() == 0.0


def test_build_s1_oracle_from_samples_uses_exact_signature_with_default_vocab():
    from finetune_tw.score_oracle import build_s1_oracle_from_samples

    samples = [
        {
            "s1_ids": torch.tensor([10, 11, 12], dtype=torch.long),
            "open_prices": torch.tensor([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0], dtype=torch.float32),
        },
        {
            "s1_ids": torch.tensor([20, 21, 12], dtype=torch.long),
            "open_prices": torch.tensor([110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0], dtype=torch.float32),
        },
    ]

    oracle = build_s1_oracle_from_samples(samples, horizon=5, min_count=2)

    assert oracle.shape == (1024,)
    assert oracle.dtype == torch.float32
