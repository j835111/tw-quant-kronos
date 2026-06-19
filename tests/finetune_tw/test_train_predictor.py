import torch

from finetune_tw.train_predictor import _resolve_amp


def test_resolve_amp_bf16():
    enabled, dtype = _resolve_amp("bf16")
    assert enabled is True
    assert dtype == torch.bfloat16


def test_resolve_amp_none():
    enabled, dtype = _resolve_amp("none")
    assert enabled is False
    assert dtype is None


def test_resolve_amp_unknown_falls_back_to_disabled():
    enabled, dtype = _resolve_amp("fp16")  # This plan does not support fp16; treat as disabled.
    assert enabled is False
    assert dtype is None
