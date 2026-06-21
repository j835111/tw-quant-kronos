import torch

from finetune_tw.train_tokenizer import _resolve_runtime_flags, _steps_for_epoch


def test_resolve_runtime_flags_enables_bf16_tf32():
    flags = _resolve_runtime_flags("bf16", enable_tf32=True)
    assert flags["amp_enabled"] is True
    assert flags["amp_dtype"] == torch.bfloat16
    assert flags["enable_tf32"] is True


def test_resolve_runtime_flags_disables_amp_for_none():
    flags = _resolve_runtime_flags("none", enable_tf32=False)
    assert flags["amp_enabled"] is False
    assert flags["amp_dtype"] is None
    assert flags["enable_tf32"] is False


def test_steps_for_epoch_uses_full_loader_when_cap_is_zero():
    assert _steps_for_epoch(loader_len=300, step_cap=0) == 300


def test_steps_for_epoch_respects_config_cap():
    assert _steps_for_epoch(loader_len=300, step_cap=120) == 120
