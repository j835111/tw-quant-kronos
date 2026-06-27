import torch
import pandas as pd
from pathlib import Path

from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices
from finetune_tw.train_predictor import (
    _build_ctx_for_date,
    _backup_predictor_checkpoint,
    _restore_predictor_training_state,
    _resolve_amp,
    _steps_for_epoch,
    _token_cache_paths,
)


def test_config_accepts_training_control_fields():
    cfg = Config(
        train_steps_per_epoch=1000,
        val_steps_per_epoch=200,
        persistent_workers=True,
        prefetch_factor=2,
        enable_tf32=True,
        token_cache_enabled=True,
        token_cache_dtype="uint16",
    )
    assert cfg.train_steps_per_epoch == 1000
    assert cfg.val_steps_per_epoch == 200
    assert cfg.persistent_workers is True
    assert cfg.prefetch_factor == 2
    assert cfg.enable_tf32 is True
    assert cfg.token_cache_enabled is True
    assert cfg.token_cache_dtype == "uint16"


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
    ctx_df, x_ts, y_ts, last_date, ctx_open = built
    assert len(ctx_df) == 90
    assert list(ctx_df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert len(x_ts) == 90
    assert len(y_ts) == cfg.pred_len
    assert last_date == x_ts.iloc[-1]
    assert ctx_open == ctx_df["open"].iloc[-1]


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


def test_token_cache_paths_are_split_specific(tmp_path):
    path = _token_cache_paths(Path(tmp_path), "train")
    assert path["data"].name == "train_token_cache.pt"
    assert path["meta"].name == "train_token_cache_meta.json"


def test_predictor_steps_for_epoch_uses_cap():
    assert _steps_for_epoch(500, 120) == 120


def test_restore_predictor_training_state_prefers_local_checkpoint(tmp_path, monkeypatch):
    exp_dir = tmp_path
    ckpt_dir = exp_dir / "predictor" / "checkpoints"
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

    monkeypatch.setattr("finetune_tw.train_predictor._gdrive_restore_checkpoints", fake_gdrive_restore)
    monkeypatch.setattr("finetune_tw.train_predictor.restore_checkpoints", fake_hf_restore)
    monkeypatch.setattr("finetune_tw.train_predictor._load_latest_checkpoint", fake_load)

    cfg = Config()
    state = _restore_predictor_training_state(
        cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        remote_root="gdrive:Kronos/outputs/test/predictor",
        model=object(),
        optimizer=object(),
        scheduler=object(),
    )

    assert state == (3, 10)
    assert calls == [
        ("gdrive", ckpt_dir, "gdrive:Kronos/outputs/test/predictor/checkpoints"),
        ("load", ckpt_dir),
    ]


def test_restore_predictor_training_state_uses_hf_when_local_missing(tmp_path, monkeypatch):
    exp_dir = tmp_path
    ckpt_dir = exp_dir / "predictor" / "checkpoints"
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

    monkeypatch.setattr("finetune_tw.train_predictor._gdrive_restore_checkpoints", fake_gdrive_restore)
    monkeypatch.setattr("finetune_tw.train_predictor.restore_checkpoints", fake_hf_restore)
    monkeypatch.setattr("finetune_tw.train_predictor._load_latest_checkpoint", fake_load)

    cfg = Config(hf_repo="org/repo", hf_checkpoint_revision_out="checkpoints-round-3")
    state = _restore_predictor_training_state(
        cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        remote_root="gdrive:Kronos/outputs/test/predictor",
        model=object(),
        optimizer=object(),
        scheduler=object(),
    )

    assert state == (4, 20)
    assert calls == [
        ("gdrive", ckpt_dir, "gdrive:Kronos/outputs/test/predictor/checkpoints"),
        ("hf", exp_dir, "org/repo", "predictor/checkpoints", "checkpoints-round-3"),
        ("load", ckpt_dir),
    ]
    assert (ckpt_dir / "ckpt-20.pt").exists()


def test_restore_predictor_training_state_ignores_invalid_local_checkpoint_entries(tmp_path, monkeypatch):
    exp_dir = tmp_path
    ckpt_dir = exp_dir / "predictor" / "checkpoints"
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

    monkeypatch.setattr("finetune_tw.train_predictor._gdrive_restore_checkpoints", fake_gdrive_restore)
    monkeypatch.setattr("finetune_tw.train_predictor.restore_checkpoints", fake_hf_restore)
    monkeypatch.setattr("finetune_tw.train_predictor._load_latest_checkpoint", fake_load)

    cfg = Config(hf_repo="org/repo", hf_checkpoint_revision_out="checkpoints-round-3")
    state = _restore_predictor_training_state(
        cfg,
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        remote_root="gdrive:Kronos/outputs/test/predictor",
        model=object(),
        optimizer=object(),
        scheduler=object(),
    )

    assert state == (5, 30)
    assert calls == [
        ("gdrive", ckpt_dir, "gdrive:Kronos/outputs/test/predictor/checkpoints"),
        ("hf", exp_dir, "org/repo", "predictor/checkpoints", "checkpoints-round-3"),
        ("load", ckpt_dir),
    ]


def test_backup_predictor_checkpoint_pushes_gdrive_and_hf_when_configured(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "predictor" / "checkpoints" / "ckpt-30.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_bytes(b"x")
    calls = []

    def fake_gdrive_sync(path, remote):
        calls.append(("gdrive", path, remote))

    def fake_push_checkpoint(local_path, repo_id, path_in_repo, revision, keep_last_n):
        calls.append(("hf", local_path, repo_id, path_in_repo, revision, keep_last_n))

    monkeypatch.setattr("finetune_tw.train_predictor._gdrive_sync_checkpoint", fake_gdrive_sync)
    monkeypatch.setattr("finetune_tw.train_predictor.push_checkpoint", fake_push_checkpoint)

    cfg = Config(
        hf_repo="org/repo",
        hf_checkpoint_revision_out="checkpoints-round-3",
        hf_checkpoint_keep_last_n=7,
    )
    _backup_predictor_checkpoint(
        cfg,
        ckpt_path=ckpt_path,
        remote_root="gdrive:Kronos/outputs/test/predictor",
    )

    assert calls == [
        ("gdrive", ckpt_path, "gdrive:Kronos/outputs/test/predictor/checkpoints"),
        (
            "hf",
            ckpt_path,
            "org/repo",
            "predictor/checkpoints/ckpt-30.pt",
            "checkpoints-round-3",
            7,
        ),
    ]
