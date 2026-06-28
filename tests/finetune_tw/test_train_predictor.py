import torch
import pandas as pd
import numpy as np
from pathlib import Path

from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices
from finetune_tw.train_predictor import (
    _build_validation_contexts,
    _build_ctx_for_date,
    _backup_predictor_checkpoint,
    _ensure_tokenizer_best_model,
    _maybe_make_predict_prepared_batch_fn,
    _run_validation_metrics,
    _restore_predictor_training_state,
    _resolve_amp,
    _steps_for_epoch,
    _token_cache_paths,
)
from model.kronos import calc_time_stamps


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


def test_build_validation_contexts_uses_single_window_query(monkeypatch):
    calls = []

    def fake_query_symbols_window(db_path, symbols, start=None, end=None):
        calls.append((db_path, symbols, start, end))
        dates = pd.bdate_range("2024-01-01", periods=5)
        rows = []
        for symbol in ["AAA", "BBB"]:
            for date in dates:
                rows.append(
                    {
                        "symbol": symbol,
                        "date": date.strftime("%Y-%m-%d"),
                        "open": 1.0,
                        "high": 2.0,
                        "low": 0.5,
                        "close": 1.5,
                        "volume": 100.0,
                        "amount": 1000.0,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr("finetune_tw.train_predictor.query_symbols_window", fake_query_symbols_window)

    cfg = Config(db_path="ignored", lookback_window=3, pred_len=2)
    val_universe = ["AAA", "BBB"]
    val_dates = pd.to_datetime(["2024-01-04", "2024-01-05"])

    contexts = _build_validation_contexts(cfg, val_universe, val_dates)

    assert len(calls) == 1
    assert calls[0][0] == cfg.db_path
    assert calls[0][1] == val_universe
    assert calls[0][2] == (
        min(val_dates) - pd.Timedelta(days=cfg.lookback_window * 2)
    ).strftime("%Y-%m-%d")
    assert calls[0][3] == max(val_dates).strftime("%Y-%m-%d")
    assert set(contexts) == set(pd.to_datetime(val_dates))
    assert [symbol for symbol, *_ in contexts[pd.Timestamp("2024-01-04")]] == ["AAA", "BBB"]
    first_context = contexts[pd.Timestamp("2024-01-04")][0]
    assert len(first_context) == 7
    _, _, x_ts, y_ts, _, x_stamp, y_stamp = first_context
    np.testing.assert_allclose(
        x_stamp,
        calc_time_stamps(x_ts).values.astype(np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        y_stamp,
        calc_time_stamps(y_ts).values.astype(np.float32),
        rtol=0,
        atol=0,
    )
    assert y_stamp is contexts[pd.Timestamp("2024-01-04")][1][6]


def test_build_validation_contexts_returns_empty_lists_when_query_is_empty(monkeypatch):
    def fake_query_symbols_window(db_path, symbols, start=None, end=None):
        return pd.DataFrame(
            columns=["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
        )

    monkeypatch.setattr("finetune_tw.train_predictor.query_symbols_window", fake_query_symbols_window)

    cfg = Config(db_path="ignored", lookback_window=3, pred_len=2)
    val_universe = ["AAA", "BBB"]
    val_dates = pd.to_datetime(["2024-01-04", "2024-01-05"])

    contexts = _build_validation_contexts(cfg, val_universe, val_dates)

    assert contexts == {pd.Timestamp(date): [] for date in val_dates}


def test_build_validation_contexts_skips_short_or_null_contexts(monkeypatch):
    def fake_query_symbols_window(db_path, symbols, start=None, end=None):
        return pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "date": "2024-01-03",
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 100.0,
                    "amount": 1000.0,
                },
                {
                    "symbol": "AAA",
                    "date": "2024-01-04",
                    "open": 1.1,
                    "high": 2.1,
                    "low": 0.6,
                    "close": 1.6,
                    "volume": 101.0,
                    "amount": 1001.0,
                },
                {
                    "symbol": "AAA",
                    "date": "2024-01-05",
                    "open": 1.2,
                    "high": 2.2,
                    "low": 0.7,
                    "close": 1.7,
                    "volume": 102.0,
                    "amount": 1002.0,
                },
                {
                    "symbol": "BBB",
                    "date": "2024-01-04",
                    "open": 3.0,
                    "high": 4.0,
                    "low": 2.5,
                    "close": 3.5,
                    "volume": 200.0,
                    "amount": 2000.0,
                },
                {
                    "symbol": "BBB",
                    "date": "2024-01-05",
                    "open": 3.1,
                    "high": 4.1,
                    "low": 2.6,
                    "close": 3.6,
                    "volume": 201.0,
                    "amount": 2001.0,
                },
                {
                    "symbol": "CCC",
                    "date": "2024-01-03",
                    "open": 5.0,
                    "high": 6.0,
                    "low": 4.5,
                    "close": 5.5,
                    "volume": 300.0,
                    "amount": 3000.0,
                },
                {
                    "symbol": "CCC",
                    "date": "2024-01-04",
                    "open": None,
                    "high": 6.1,
                    "low": 4.6,
                    "close": 5.6,
                    "volume": 301.0,
                    "amount": 3001.0,
                },
                {
                    "symbol": "CCC",
                    "date": "2024-01-05",
                    "open": 5.2,
                    "high": 6.2,
                    "low": 4.7,
                    "close": 5.7,
                    "volume": 302.0,
                    "amount": 3002.0,
                },
            ]
        )

    monkeypatch.setattr("finetune_tw.train_predictor.query_symbols_window", fake_query_symbols_window)

    cfg = Config(db_path="ignored", lookback_window=3, pred_len=2)
    val_universe = ["AAA", "BBB", "CCC"]
    val_dates = [pd.Timestamp("2024-01-05")]

    contexts = _build_validation_contexts(cfg, val_universe, val_dates)

    assert [symbol for symbol, *_ in contexts[pd.Timestamp("2024-01-05")]] == ["AAA"]


def test_training_validation_path_builds_contexts_once_and_reuses_rows(monkeypatch):
    cfg = Config(pred_len=3, lookback_window=3)
    val_dates = pd.to_datetime(["2024-01-03", "2024-01-04"])
    contexts_by_date = {
        pd.Timestamp("2024-01-03"): [("AAA", object(), object(), object(), pd.Timestamp("2024-01-02"))],
        pd.Timestamp("2024-01-04"): [("BBB", object(), object(), object(), pd.Timestamp("2024-01-03"))],
    }
    rows_by_date = {
        pd.Timestamp("2024-01-03"): [("AAA", np.array([1.0, 2.0, 3.0]), 1.0, pd.Timestamp("2024-01-02"))],
        pd.Timestamp("2024-01-04"): [("BBB", np.array([1.5, 2.5, 3.5]), 1.5, pd.Timestamp("2024-01-03"))],
    }
    calls = {"build": [], "collect": [], "metrics": []}

    def fake_build_validation_contexts(cfg_arg, val_universe_arg, val_dates_arg):
        calls["build"].append((cfg_arg, list(val_universe_arg), list(val_dates_arg)))
        return contexts_by_date

    def fake_collect_validation_rows_by_date(
        predict_batch_fn,
        contexts_by_date_arg,
        cfg_arg,
        batch_size=64,
        prepared_batch_predict_fn=None,
    ):
        calls["collect"].append(
            (predict_batch_fn, contexts_by_date_arg, cfg_arg, batch_size, prepared_batch_predict_fn)
        )
        return rows_by_date

    def fake_compute_validation_metrics_from_rows(
        rows_by_date_arg,
        actual_lookup_arg,
        val_dates_arg,
        cfg_arg,
        target_horizon,
        compute_ic=True,
        compute_ic_ir=True,
    ):
        calls["metrics"].append(
            (
                rows_by_date_arg,
                actual_lookup_arg,
                list(val_dates_arg),
                cfg_arg,
                target_horizon,
                compute_ic,
                compute_ic_ir,
            )
        )
        return 0.5, 0.4

    monkeypatch.setattr(
        "finetune_tw.train_predictor._build_validation_contexts",
        fake_build_validation_contexts,
    )
    monkeypatch.setattr(
        "finetune_tw.train_predictor.collect_validation_rows_by_date",
        fake_collect_validation_rows_by_date,
    )
    monkeypatch.setattr(
        "finetune_tw.train_predictor.compute_validation_metrics_from_rows",
        fake_compute_validation_metrics_from_rows,
    )

    result = _run_validation_metrics(
        cfg=cfg,
        predict_batch_fn=object(),
        prepared_batch_predict_fn=object(),
        actual_lookup=lambda sym, last_date, n: np.array([1.0, 2.0, 3.0], dtype=float),
        val_universe=["AAA", "BBB"],
        val_dates=val_dates,
    )

    assert result == (0.5, 0.4)
    assert len(calls["build"]) == 1
    assert len(calls["collect"]) == 1
    assert len(calls["metrics"]) == 1


def test_run_validation_metrics_uses_prebuilt_contexts_without_rebuilding(monkeypatch):
    cfg = Config(pred_len=3, lookback_window=3)
    val_dates = pd.to_datetime(["2024-01-03", "2024-01-04"])
    contexts_by_date = {
        pd.Timestamp("2024-01-03"): [("AAA", object(), object(), object(), pd.Timestamp("2024-01-02"))],
        pd.Timestamp("2024-01-04"): [("BBB", object(), object(), object(), pd.Timestamp("2024-01-03"))],
    }
    rows_by_date = {
        pd.Timestamp("2024-01-03"): [("AAA", np.array([1.0, 2.0, 3.0]), 1.0, pd.Timestamp("2024-01-02"))],
        pd.Timestamp("2024-01-04"): [("BBB", np.array([1.5, 2.5, 3.5]), 1.5, pd.Timestamp("2024-01-03"))],
    }
    calls = {"collect": [], "metrics": []}

    def fail_build_validation_contexts(*args, **kwargs):
        raise AssertionError("_build_validation_contexts should not be called when contexts_by_date is provided")

    def fake_collect_validation_rows_by_date(
        predict_batch_fn,
        contexts_by_date_arg,
        cfg_arg,
        batch_size=64,
        prepared_batch_predict_fn=None,
    ):
        calls["collect"].append(
            (predict_batch_fn, contexts_by_date_arg, cfg_arg, batch_size, prepared_batch_predict_fn)
        )
        return rows_by_date

    def fake_compute_validation_metrics_from_rows(
        rows_by_date_arg,
        actual_lookup_arg,
        val_dates_arg,
        cfg_arg,
        target_horizon,
        compute_ic=True,
        compute_ic_ir=True,
    ):
        calls["metrics"].append(
            (
                rows_by_date_arg,
                actual_lookup_arg,
                list(val_dates_arg),
                cfg_arg,
                target_horizon,
                compute_ic,
                compute_ic_ir,
            )
        )
        return 0.6, 0.3

    monkeypatch.setattr(
        "finetune_tw.train_predictor._build_validation_contexts",
        fail_build_validation_contexts,
    )
    monkeypatch.setattr(
        "finetune_tw.train_predictor.collect_validation_rows_by_date",
        fake_collect_validation_rows_by_date,
    )
    monkeypatch.setattr(
        "finetune_tw.train_predictor.compute_validation_metrics_from_rows",
        fake_compute_validation_metrics_from_rows,
    )

    result = _run_validation_metrics(
        cfg=cfg,
        predict_batch_fn=object(),
        prepared_batch_predict_fn=None,
        actual_lookup=lambda sym, last_date, n: np.array([1.0, 2.0, 3.0], dtype=float),
        val_universe=["AAA", "BBB"],
        val_dates=val_dates,
        contexts_by_date=contexts_by_date,
    )

    assert result == (0.6, 0.3)
    assert len(calls["collect"]) == 1
    assert calls["collect"][0][1] is contexts_by_date
    assert len(calls["metrics"]) == 1


def test_maybe_make_predict_prepared_batch_fn_returns_none_for_legacy_predictor():
    class LegacyPredictor:
        def predict_batch(self, *args, **kwargs):
            return []

    assert _maybe_make_predict_prepared_batch_fn(LegacyPredictor()) is None


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


def test_ensure_tokenizer_best_model_restores_from_hf_when_missing(tmp_path, monkeypatch):
    exp_dir = tmp_path
    calls = []

    def fake_restore_best_model(exp_dir_arg, repo_id, subfolder, revision):
        calls.append((exp_dir_arg, repo_id, subfolder, revision))
        target = exp_dir_arg / subfolder
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"x")
        return True

    monkeypatch.setattr("finetune_tw.train_predictor.restore_best_model", fake_restore_best_model)

    cfg = Config(hf_repo="org/repo", hf_revision_out="round-3")
    path = _ensure_tokenizer_best_model(cfg, exp_dir)

    assert path == exp_dir / "tokenizer" / "best_model"
    assert calls == [
        (exp_dir, "org/repo", "tokenizer/best_model", "round-3"),
    ]
