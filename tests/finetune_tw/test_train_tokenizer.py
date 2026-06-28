import torch

from finetune_tw.config import Config
from finetune_tw.train_tokenizer import (
    _load_latest_checkpoint,
    _backup_tokenizer_checkpoint,
    _resolve_runtime_flags,
    _restore_tokenizer_training_state,
    _steps_for_epoch,
)


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


def test_restore_tokenizer_training_state_prefers_local_checkpoint(tmp_path, monkeypatch):
    exp_dir = tmp_path
    ckpt_dir = exp_dir / "tokenizer" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-10.pt").write_bytes(b"x")
    calls = []

    def fake_gdrive_restore(path, remote):
        calls.append(("gdrive", path, remote))

    def fake_hf_restore(*args, **kwargs):
        calls.append(("hf", args, kwargs))
        return 1

    def fake_load(path, model, optimizer, scheduler):
        calls.append(("load", path))
        return 3, 10

    monkeypatch.setattr("finetune_tw.train_tokenizer._gdrive_restore_checkpoints", fake_gdrive_restore)
    monkeypatch.setattr("finetune_tw.train_tokenizer.restore_checkpoints", fake_hf_restore)
    monkeypatch.setattr("finetune_tw.train_tokenizer._load_latest_checkpoint", fake_load)

    cfg = Config()
    state = _restore_tokenizer_training_state(
        cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        remote_root="gdrive:Kronos/outputs/test/tokenizer",
        model=object(),
        optimizer=object(),
        scheduler=object(),
    )

    assert state == (3, 10)
    assert calls == [
        ("gdrive", ckpt_dir, "gdrive:Kronos/outputs/test/tokenizer/checkpoints"),
        ("load", ckpt_dir),
    ]


def test_restore_tokenizer_training_state_uses_hf_when_local_missing(tmp_path, monkeypatch):
    exp_dir = tmp_path
    ckpt_dir = exp_dir / "tokenizer" / "checkpoints"
    calls = []

    def fake_gdrive_restore(path, remote):
        calls.append(("gdrive", path, remote))

    def fake_hf_restore(exp_dir, repo_id, subfolder, revision):
        calls.append(("hf", exp_dir, repo_id, subfolder, revision))
        target = exp_dir / subfolder
        target.mkdir(parents=True, exist_ok=True)
        (target / "ckpt-20.pt").write_bytes(b"x")
        return 1

    def fake_load(path, model, optimizer, scheduler):
        calls.append(("load", path))
        return 4, 20

    monkeypatch.setattr("finetune_tw.train_tokenizer._gdrive_restore_checkpoints", fake_gdrive_restore)
    monkeypatch.setattr("finetune_tw.train_tokenizer.restore_checkpoints", fake_hf_restore)
    monkeypatch.setattr("finetune_tw.train_tokenizer._load_latest_checkpoint", fake_load)

    cfg = Config(hf_repo="org/repo", hf_checkpoint_revision_out="checkpoints-round-3")
    state = _restore_tokenizer_training_state(
        cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        remote_root="gdrive:Kronos/outputs/test/tokenizer",
        model=object(),
        optimizer=object(),
        scheduler=object(),
    )

    assert state == (4, 20)
    assert calls == [
        ("gdrive", ckpt_dir, "gdrive:Kronos/outputs/test/tokenizer/checkpoints"),
        ("hf", exp_dir, "org/repo", "tokenizer/checkpoints", "checkpoints-round-3"),
        ("load", ckpt_dir),
    ]
    assert (ckpt_dir / "ckpt-20.pt").exists()


def test_restore_tokenizer_training_state_ignores_invalid_local_checkpoint_entries(tmp_path, monkeypatch):
    exp_dir = tmp_path
    ckpt_dir = exp_dir / "tokenizer" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-10.pt").write_bytes(b"")
    (ckpt_dir / "ckpt-latest.pt").write_bytes(b"x")
    calls = []

    def fake_gdrive_restore(path, remote):
        calls.append(("gdrive", path, remote))

    def fake_hf_restore(exp_dir, repo_id, subfolder, revision):
        calls.append(("hf", exp_dir, repo_id, subfolder, revision))
        return 1

    def fake_load(path, model, optimizer, scheduler):
        calls.append(("load", path))
        return 5, 30

    monkeypatch.setattr("finetune_tw.train_tokenizer._gdrive_restore_checkpoints", fake_gdrive_restore)
    monkeypatch.setattr("finetune_tw.train_tokenizer.restore_checkpoints", fake_hf_restore)
    monkeypatch.setattr("finetune_tw.train_tokenizer._load_latest_checkpoint", fake_load)

    cfg = Config(hf_repo="org/repo", hf_checkpoint_revision_out="checkpoints-round-3")
    state = _restore_tokenizer_training_state(
        cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        remote_root="gdrive:Kronos/outputs/test/tokenizer",
        model=object(),
        optimizer=object(),
        scheduler=object(),
    )

    assert state == (5, 30)
    assert calls == [
        ("gdrive", ckpt_dir, "gdrive:Kronos/outputs/test/tokenizer/checkpoints"),
        ("hf", exp_dir, "org/repo", "tokenizer/checkpoints", "checkpoints-round-3"),
        ("load", ckpt_dir),
    ]


def test_backup_tokenizer_checkpoint_pushes_gdrive_and_hf_when_configured(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "tokenizer" / "checkpoints" / "ckpt-30.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"x")
    calls = []

    def fake_gdrive_sync(path, remote):
        calls.append(("gdrive", path, remote))

    def fake_push_checkpoint(local_path, repo_id, path_in_repo, revision, keep_last_n):
        calls.append(("hf", local_path, repo_id, path_in_repo, revision, keep_last_n))

    monkeypatch.setattr("finetune_tw.train_tokenizer._gdrive_sync_checkpoint", fake_gdrive_sync)
    monkeypatch.setattr("finetune_tw.train_tokenizer.push_checkpoint", fake_push_checkpoint)

    cfg = Config(
        hf_repo="org/repo",
        hf_checkpoint_revision_out="checkpoints-round-3",
        hf_checkpoint_keep_last_n=7,
    )
    _backup_tokenizer_checkpoint(
        cfg,
        ckpt_path=ckpt_path,
        remote_root="gdrive:Kronos/outputs/test/tokenizer",
    )

    assert calls == [
        ("gdrive", ckpt_path, "gdrive:Kronos/outputs/test/tokenizer/checkpoints"),
        (
            "hf",
            ckpt_path,
            "org/repo",
            "tokenizer/checkpoints/ckpt-30.pt",
            "checkpoints-round-3",
            7,
        ),
    ]


def test_load_latest_checkpoint_skips_corrupt_newest_file(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "tokenizer" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-10.pt").write_bytes(b"corrupt")
    good = {
        "step": 9,
        "epoch": 2,
        "model": {"m": 1},
        "optimizer": {"o": 2},
        "scheduler": {"s": 3},
    }
    (ckpt_dir / "ckpt-9.pt").write_bytes(b"x")

    calls = []

    class FakeObj:
        def __init__(self, key):
            self.key = key

        def load_state_dict(self, payload):
            calls.append((self.key, payload))

    def fake_torch_load(path, map_location=None, weights_only=None):
        if path.name == "ckpt-10.pt":
            raise RuntimeError("bad checkpoint")
        return good

    monkeypatch.setattr("finetune_tw.train_tokenizer.torch.load", fake_torch_load)

    epoch, step = _load_latest_checkpoint(
        ckpt_dir,
        FakeObj("model"),
        FakeObj("optimizer"),
        FakeObj("scheduler"),
    )

    assert (epoch, step) == (2, 9)
    assert calls == [
        ("model", {"m": 1}),
        ("optimizer", {"o": 2}),
        ("scheduler", {"s": 3}),
    ]
