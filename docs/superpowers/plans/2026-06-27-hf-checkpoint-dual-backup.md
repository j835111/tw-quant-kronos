# HF Checkpoint Dual Backup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move recoverable fine-tuning state to `/mnt/first/kronos_state`, add HF checkpoint backup and restore on `checkpoints-round-3`, and add a one-command MoLab resume script that can rebuild `/marimo/Kronos` after sandbox loss.

**Architecture:** Treat `/marimo/Kronos` as disposable code checkout only and move DB, outputs, logs, and pid files under `/mnt/first/kronos_state`. Training restores checkpoints in this order: local checkpoint directory, HF checkpoint branch, then the existing pretrained or best-model path. HF checkpoint uploads stay asynchronous and best-effort, and a new shell bootstrap script recreates the repo checkout, validates config paths, launches the selected stage, and restarts a monitor whose files also live under the state directory.

**Tech Stack:** Python 3.10+, PyTorch, `huggingface_hub==0.33.1`, bash, pytest

---

## File Map

- Modify: `finetune_tw/config.py`
  - Add HF checkpoint backup settings used by both training stages.
- Modify: `finetune_tw/configs/config_tw_daily_rtx6000.yaml`
  - Move DB and outputs to `/mnt/first/kronos_state` and set the checkpoint branch.
- Modify: `finetune_tw/hf_utils.py`
  - Add local checkpoint listing, HF checkpoint restore, remote prune, and background checkpoint upload helpers.
- Modify: `finetune_tw/train_predictor.py`
  - Add local-first/HF-fallback checkpoint restore and HF checkpoint backup hooks.
- Modify: `finetune_tw/train_tokenizer.py`
  - Mirror the predictor checkpoint behavior so tokenizer and predictor recover the same way.
- Create: `scripts/resume_molab_training.sh`
  - Rebuild repo checkout, validate config paths, launch training, and start monitoring.
- Modify: `tests/finetune_tw/test_config_retrain.py`
  - Cover new config fields and the RTX6000 persistent-state YAML.
- Create: `tests/finetune_tw/test_hf_utils.py`
  - Cover restore, prune, and upload failure behavior for HF checkpoints.
- Modify: `tests/finetune_tw/test_train_predictor.py`
  - Cover predictor checkpoint restore and backup helpers.
- Modify: `tests/finetune_tw/test_train_tokenizer.py`
  - Cover tokenizer checkpoint restore and backup helpers.
- Create: `tests/finetune_tw/test_resume_molab_training_sh.py`
  - Cover repo bootstrap, state bootstrap, and config-path safety checks.

### Task 1: Config Schema And Persistent-State Baseline

**Files:**
- Modify: `finetune_tw/config.py`
- Modify: `finetune_tw/configs/config_tw_daily_rtx6000.yaml`
- Test: `tests/finetune_tw/test_config_retrain.py`

- [ ] **Step 1: Write the failing tests**

```python
from finetune_tw.config import Config


def test_config_defaults_include_hf_checkpoint_fields():
    cfg = Config()
    assert cfg.hf_checkpoint_revision_out == ""
    assert cfg.hf_checkpoint_keep_last_n == 3


def test_rtx6000_yaml_uses_persistent_state_paths():
    cfg = Config.from_yaml("finetune_tw/configs/config_tw_daily_rtx6000.yaml")
    assert cfg.db_path == "/mnt/first/kronos_state/data/tw_stocks.db"
    assert cfg.output_dir == "/mnt/first/kronos_state/outputs"
    assert cfg.hf_checkpoint_revision_out == "checkpoints-round-3"
    assert cfg.hf_checkpoint_keep_last_n == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/finetune_tw/test_config_retrain.py -k "hf_checkpoint or rtx6000" -v`

Expected: FAIL because `Config` does not define `hf_checkpoint_revision_out` or `hf_checkpoint_keep_last_n`, and the RTX6000 config still points to `/marimo/Kronos`.

- [ ] **Step 3: Write the minimal implementation**

```python
@dataclass
class Config:
    # ...
    hf_repo: str = ""
    hf_revision: str = ""
    hf_revision_out: str = ""
    hf_checkpoint_revision_out: str = ""
    hf_checkpoint_keep_last_n: int = 3
```

```yaml
db_path: "/mnt/first/kronos_state/data/tw_stocks.db"
lookback_window: 90
predict_window: 10
max_context: 512
clip: 5.0
train_end_date: "2023-12-31"
val_end_date: "2024-06-30"

tokenizer_epochs: 12
basemodel_epochs: 20
batch_size: 256
save_steps: 500
log_interval: 50
tokenizer_lr: 0.0002
predictor_lr: 0.00005
adam_beta1: 0.9
adam_beta2: 0.95
adam_weight_decay: 0.1
num_workers: 4
persistent_workers: true
prefetch_factor: 2
train_steps_per_epoch: 1000
val_steps_per_epoch: 200
amp_dtype: "bf16"
enable_tf32: true
token_cache_enabled: true
token_cache_dtype: "uint16"
seed: 42

early_stop_patience: 3
ic_val_symbols: 500
ic_val_dates: 40
ranking_loss_alpha: 0.0
ranking_loss_horizon: 5

pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
pretrained_predictor: "j835111/kronos-tw-finetune"
hf_revision: "round-0"
hf_repo: "j835111/kronos-tw-finetune"
hf_revision_out: "round-3"
hf_checkpoint_revision_out: "checkpoints-round-3"
hf_checkpoint_keep_last_n: 3
exp_name: "tw_daily"
output_dir: "/mnt/first/kronos_state/outputs"

top_k: 10
hold_days: 5
pred_len: 10
test_start_date: "2024-07-01"
benchmark_symbol: "^TWII"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/finetune_tw/test_config_retrain.py -k "hf_checkpoint or rtx6000" -v`

Expected: PASS for both new config assertions.

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/config.py finetune_tw/configs/config_tw_daily_rtx6000.yaml tests/finetune_tw/test_config_retrain.py
git commit -m "feat: move rtx6000 training state to persistent storage"
```

### Task 2: HF Checkpoint Helpers

**Files:**
- Modify: `finetune_tw/hf_utils.py`
- Create: `tests/finetune_tw/test_hf_utils.py`

- [ ] **Step 1: Write the failing tests**

```python
import sys
import types
from pathlib import Path

import finetune_tw.hf_utils as hf_utils


class FakeRepoFile:
    def __init__(self, path: str) -> None:
        self.path = path


class FakeHfApi:
    def __init__(self, tree: list[str] | None = None) -> None:
        self.tree = tree or []
        self.deleted: list[tuple[str, str | None]] = []
        self.branches: list[tuple[str, str, bool]] = []

    def create_branch(self, repo_id: str, *, branch: str, token: str | None = None, exist_ok: bool = False, revision: str | None = None, repo_type: str | None = None) -> None:
        self.branches.append((repo_id, branch, exist_ok))

    def list_repo_tree(self, repo_id: str, path_in_repo: str | None = None, *, recursive: bool = False, expand: bool = False, revision: str | None = None, repo_type: str | None = None, token: str | None = None):
        return [FakeRepoFile(path) for path in self.tree]

    def delete_file(self, path_in_repo: str, repo_id: str, *, revision: str | None = None, token: str | None = None, repo_type: str | None = None, commit_message: str | None = None, commit_description: str | None = None, create_pr: bool | None = None, parent_commit: str | None = None) -> None:
        self.deleted.append((path_in_repo, revision))


def test_restore_checkpoints_skips_remote_when_local_exists(tmp_path, monkeypatch):
    exp_dir = tmp_path / "tw_daily"
    ckpt_dir = exp_dir / "predictor" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-500.pt").write_bytes(b"x")
    called: dict[str, object] = {}
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda **kwargs: called.setdefault("download", kwargs)))

    restored = hf_utils.restore_checkpoints(exp_dir, "repo", "predictor/checkpoints", "checkpoints-round-3")

    assert restored == 0
    assert "download" not in called


def test_restore_checkpoints_downloads_when_local_missing(tmp_path, monkeypatch):
    exp_dir = tmp_path / "tw_daily"
    called: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        called["kwargs"] = kwargs
        target = Path(kwargs["local_dir"]) / "predictor" / "checkpoints"
        target.mkdir(parents=True, exist_ok=True)
        (target / "ckpt-700.pt").write_bytes(b"x")
        return str(kwargs["local_dir"])

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))

    restored = hf_utils.restore_checkpoints(exp_dir, "repo", "predictor/checkpoints", "checkpoints-round-3")

    assert restored == 1
    assert called["kwargs"]["allow_patterns"] == ["predictor/checkpoints/ckpt-*.pt"]


def test_prune_checkpoints_keeps_latest_three(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    api = FakeHfApi(
        tree=[
            "predictor/checkpoints/ckpt-100.pt",
            "predictor/checkpoints/ckpt-200.pt",
            "predictor/checkpoints/ckpt-300.pt",
            "predictor/checkpoints/ckpt-400.pt",
            "predictor/checkpoints/ckpt-500.pt",
        ]
    )

    stale = hf_utils.prune_checkpoints("repo", "predictor/checkpoints", "checkpoints-round-3", 3, api=api)

    assert stale == [
        "predictor/checkpoints/ckpt-100.pt",
        "predictor/checkpoints/ckpt-200.pt",
    ]
    assert api.deleted == [
        ("predictor/checkpoints/ckpt-100.pt", "checkpoints-round-3"),
        ("predictor/checkpoints/ckpt-200.pt", "checkpoints-round-3"),
    ]


def test_push_checkpoint_upload_failure_is_swallowed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "token")
    ckpt = tmp_path / "ckpt-500.pt"
    ckpt.write_bytes(b"x")
    fake_api = FakeHfApi()

    def fake_upload_file(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: fake_api, upload_file=fake_upload_file),
    )

    hf_utils.push_checkpoint(
        ckpt,
        "repo",
        "predictor/checkpoints/ckpt-500.pt",
        "checkpoints-round-3",
        3,
    )
    hf_utils.wait_for_pushes()

    captured = capsys.readouterr()
    assert "checkpoint push failed" in captured.out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/finetune_tw/test_hf_utils.py -v`

Expected: FAIL because `restore_checkpoints`, `prune_checkpoints`, and `push_checkpoint` do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from pathlib import Path, PurePosixPath


def _checkpoint_step(path: Path | str) -> int:
    name = Path(path).name
    return int(name.removeprefix("ckpt-").removesuffix(".pt"))


def local_checkpoints(ckpt_dir: Path) -> list[Path]:
    return sorted(ckpt_dir.glob("ckpt-*.pt"), key=_checkpoint_step)


def restore_checkpoints(exp_dir: Path, repo_id: str, subfolder: str, revision: str) -> int:
    ckpt_dir = exp_dir / subfolder
    if local_checkpoints(ckpt_dir):
        return 0
    token = _token()
    print(f"  [hf] restoring {subfolder} from {repo_id}@{revision} ...")
    try:
        from huggingface_hub import snapshot_download

        exp_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[f"{subfolder}/ckpt-*.pt"],
            local_dir=str(exp_dir),
            token=token,
        )
        restored = len(local_checkpoints(ckpt_dir))
        if restored:
            print(f"  [hf] restored {restored} checkpoint(s) from {subfolder}")
        else:
            print(f"  [hf] no checkpoints found at {repo_id}@{revision}/{subfolder}")
        return restored
    except Exception as exc:
        print(f"  [hf] checkpoint restore failed: {exc}")
        return 0


def prune_checkpoints(repo_id: str, subfolder: str, revision: str, keep_last_n: int, api=None) -> list[str]:
    token = _token()
    if not token:
        return []
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    files = sorted(
        [
            item.path
            for item in api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=subfolder,
                revision=revision,
                token=token,
                recursive=False,
            )
            if item.path.endswith(".pt")
        ],
        key=_checkpoint_step,
    )
    stale = files[:-keep_last_n] if keep_last_n > 0 else files
    for path_in_repo in stale:
        try:
            api.delete_file(
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                revision=revision,
                token=token,
                commit_message=f"prune {PurePosixPath(path_in_repo).name}",
            )
        except Exception as exc:
            print(f"  [hf] checkpoint prune failed: {exc}")
    return stale


def push_checkpoint(local_path: Path, repo_id: str, path_in_repo: str, revision: str, keep_last_n: int) -> None:
    token = _token()
    if not token:
        print(f"  [hf] HF_TOKEN not set - skipping checkpoint push to {repo_id}@{revision}/{path_in_repo}")
        return

    def _run() -> None:
        try:
            from huggingface_hub import HfApi, upload_file

            api = HfApi()
            api.create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                revision=revision,
                token=token,
                commit_message=f"update {path_in_repo}",
            )
            prune_checkpoints(
                repo_id,
                str(PurePosixPath(path_in_repo).parent),
                revision,
                keep_last_n,
                api=api,
            )
        except Exception as exc:
            print(f"  [hf] checkpoint push failed: {exc}")

    t = threading.Thread(target=_run, daemon=False, name="hf-push-checkpoint")
    _push_threads.append(t)
    t.start()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/finetune_tw/test_hf_utils.py -v`

Expected: PASS, including the non-fatal upload failure case.

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/hf_utils.py tests/finetune_tw/test_hf_utils.py
git commit -m "feat: add hf checkpoint backup helpers"
```

### Task 3: Predictor Checkpoint Restore And Backup Integration

**Files:**
- Modify: `finetune_tw/train_predictor.py`
- Modify: `tests/finetune_tw/test_train_predictor.py`

- [ ] **Step 1: Write the failing tests**

```python
import finetune_tw.train_predictor as train_predictor
from finetune_tw.config import Config


def test_restore_predictor_training_state_prefers_local_checkpoint(tmp_path, monkeypatch):
    exp_root = tmp_path / "outputs" / "tw_daily"
    ckpt_dir = exp_root / "predictor" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-500.pt").write_bytes(b"x")
    calls: list[str] = []
    monkeypatch.setattr(train_predictor, "_gdrive_restore_checkpoints", lambda *args, **kwargs: calls.append("gdrive"))
    monkeypatch.setattr(train_predictor, "restore_checkpoints", lambda *args, **kwargs: calls.append("hf"))
    monkeypatch.setattr(train_predictor, "_load_latest_checkpoint", lambda *args, **kwargs: (4, 500))

    epoch, step = train_predictor._restore_predictor_training_state(
        Config(exp_name="tw_daily"),
        exp_root,
        ckpt_dir,
        object(),
        object(),
        object(),
    )

    assert (epoch, step) == (4, 500)
    assert calls == ["gdrive"]


def test_restore_predictor_training_state_uses_hf_fallback_when_local_missing(tmp_path, monkeypatch):
    exp_root = tmp_path / "outputs" / "tw_daily"
    ckpt_dir = exp_root / "predictor" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    calls: list[str] = []
    monkeypatch.setattr(train_predictor, "_gdrive_restore_checkpoints", lambda *args, **kwargs: calls.append("gdrive"))

    def fake_restore(exp_dir, repo_id, subfolder, revision):
        calls.append("hf")
        target = exp_dir / subfolder
        target.mkdir(parents=True, exist_ok=True)
        (target / "ckpt-900.pt").write_bytes(b"x")
        return 1

    monkeypatch.setattr(train_predictor, "restore_checkpoints", fake_restore)
    monkeypatch.setattr(train_predictor, "_load_latest_checkpoint", lambda *args, **kwargs: (7, 900))

    epoch, step = train_predictor._restore_predictor_training_state(
        Config(
            exp_name="tw_daily",
            hf_repo="repo",
            hf_checkpoint_revision_out="checkpoints-round-3",
        ),
        exp_root,
        ckpt_dir,
        object(),
        object(),
        object(),
    )

    assert (epoch, step) == (7, 900)
    assert calls == ["gdrive", "hf"]


def test_backup_predictor_checkpoint_pushes_hf_when_configured(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "predictor" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-1200.pt").write_bytes(b"x")
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(train_predictor, "_gdrive_sync_checkpoint", lambda *args, **kwargs: calls.append(("gdrive", args, kwargs)))
    monkeypatch.setattr(train_predictor, "push_checkpoint", lambda *args, **kwargs: calls.append(("hf", args, kwargs)))

    train_predictor._backup_predictor_checkpoint(
        Config(
            exp_name="tw_daily",
            hf_repo="repo",
            hf_checkpoint_revision_out="checkpoints-round-3",
            hf_checkpoint_keep_last_n=3,
        ),
        ckpt_dir,
        1200,
    )

    assert [name for name, _, _ in calls] == ["gdrive", "hf"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/finetune_tw/test_train_predictor.py -k "restore_predictor_training_state or backup_predictor_checkpoint" -v`

Expected: FAIL because `_restore_predictor_training_state` and `_backup_predictor_checkpoint` do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from finetune_tw.hf_utils import (
    push_best_model,
    push_checkpoint,
    push_file,
    restore_checkpoints,
    wait_for_pushes,
)


def _restore_predictor_training_state(cfg: Config, exp_root: Path, ckpt_dir: Path, model, optimizer, scheduler):
    remote_ckpt_dir = f"gdrive:Kronos/outputs/{cfg.exp_name}/predictor/checkpoints"
    _gdrive_restore_checkpoints(ckpt_dir, remote_ckpt_dir)
    if not list(ckpt_dir.glob("ckpt-*.pt")) and cfg.hf_repo and cfg.hf_checkpoint_revision_out:
        restore_checkpoints(exp_root, cfg.hf_repo, "predictor/checkpoints", cfg.hf_checkpoint_revision_out)
    return _load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler)


def _backup_predictor_checkpoint(cfg: Config, ckpt_dir: Path, global_step: int) -> None:
    ckpt_path = ckpt_dir / f"ckpt-{global_step}.pt"
    _gdrive_sync_checkpoint(ckpt_path, f"gdrive:Kronos/outputs/{cfg.exp_name}/predictor/checkpoints")
    if cfg.hf_repo and cfg.hf_checkpoint_revision_out:
        push_checkpoint(
            ckpt_path,
            cfg.hf_repo,
            f"predictor/checkpoints/{ckpt_path.name}",
            cfg.hf_checkpoint_revision_out,
            cfg.hf_checkpoint_keep_last_n,
        )
```

```python
exp_root = Path(cfg.output_dir) / cfg.exp_name
save_dir = exp_root / "predictor"
ckpt_dir = save_dir / "checkpoints"

start_epoch, global_step = _restore_predictor_training_state(
    cfg,
    exp_root,
    ckpt_dir,
    model,
    optimizer,
    scheduler,
)

if global_step % cfg.save_steps == 0:
    _save_checkpoint(ckpt_dir, global_step, epoch, model, optimizer, scheduler)
    _backup_predictor_checkpoint(cfg, ckpt_dir, global_step)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/finetune_tw/test_train_predictor.py -k "restore_predictor_training_state or backup_predictor_checkpoint" -v`

Expected: PASS, with local-first restore and HF backup both covered.

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/train_predictor.py tests/finetune_tw/test_train_predictor.py
git commit -m "feat: add predictor hf checkpoint fallback"
```

### Task 4: Tokenizer Checkpoint Restore And Backup Integration

**Files:**
- Modify: `finetune_tw/train_tokenizer.py`
- Modify: `tests/finetune_tw/test_train_tokenizer.py`

- [ ] **Step 1: Write the failing tests**

```python
import finetune_tw.train_tokenizer as train_tokenizer
from finetune_tw.config import Config


def test_restore_tokenizer_training_state_prefers_local_checkpoint(tmp_path, monkeypatch):
    exp_root = tmp_path / "outputs" / "tw_daily"
    ckpt_dir = exp_root / "tokenizer" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-400.pt").write_bytes(b"x")
    calls: list[str] = []
    monkeypatch.setattr(train_tokenizer, "_gdrive_restore_checkpoints", lambda *args, **kwargs: calls.append("gdrive"))
    monkeypatch.setattr(train_tokenizer, "restore_checkpoints", lambda *args, **kwargs: calls.append("hf"))
    monkeypatch.setattr(train_tokenizer, "_load_latest_checkpoint", lambda *args, **kwargs: (2, 400))

    epoch, step = train_tokenizer._restore_tokenizer_training_state(
        Config(exp_name="tw_daily"),
        exp_root,
        ckpt_dir,
        object(),
        object(),
        object(),
    )

    assert (epoch, step) == (2, 400)
    assert calls == ["gdrive"]


def test_restore_tokenizer_training_state_uses_hf_fallback_when_local_missing(tmp_path, monkeypatch):
    exp_root = tmp_path / "outputs" / "tw_daily"
    ckpt_dir = exp_root / "tokenizer" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    calls: list[str] = []
    monkeypatch.setattr(train_tokenizer, "_gdrive_restore_checkpoints", lambda *args, **kwargs: calls.append("gdrive"))

    def fake_restore(exp_dir, repo_id, subfolder, revision):
        calls.append("hf")
        target = exp_dir / subfolder
        target.mkdir(parents=True, exist_ok=True)
        (target / "ckpt-800.pt").write_bytes(b"x")
        return 1

    monkeypatch.setattr(train_tokenizer, "restore_checkpoints", fake_restore)
    monkeypatch.setattr(train_tokenizer, "_load_latest_checkpoint", lambda *args, **kwargs: (6, 800))

    epoch, step = train_tokenizer._restore_tokenizer_training_state(
        Config(
            exp_name="tw_daily",
            hf_repo="repo",
            hf_checkpoint_revision_out="checkpoints-round-3",
        ),
        exp_root,
        ckpt_dir,
        object(),
        object(),
        object(),
    )

    assert (epoch, step) == (6, 800)
    assert calls == ["gdrive", "hf"]


def test_backup_tokenizer_checkpoint_pushes_hf_when_configured(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "tokenizer" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-1000.pt").write_bytes(b"x")
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(train_tokenizer, "_gdrive_sync_checkpoint", lambda *args, **kwargs: calls.append(("gdrive", args, kwargs)))
    monkeypatch.setattr(train_tokenizer, "push_checkpoint", lambda *args, **kwargs: calls.append(("hf", args, kwargs)))

    train_tokenizer._backup_tokenizer_checkpoint(
        Config(
            exp_name="tw_daily",
            hf_repo="repo",
            hf_checkpoint_revision_out="checkpoints-round-3",
            hf_checkpoint_keep_last_n=3,
        ),
        ckpt_dir,
        1000,
    )

    assert [name for name, _, _ in calls] == ["gdrive", "hf"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/finetune_tw/test_train_tokenizer.py -k "restore_tokenizer_training_state or backup_tokenizer_checkpoint" -v`

Expected: FAIL because `_restore_tokenizer_training_state` and `_backup_tokenizer_checkpoint` do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from finetune_tw.hf_utils import push_checkpoint, restore_checkpoints


def _restore_tokenizer_training_state(cfg: Config, exp_root: Path, ckpt_dir: Path, model, optimizer, scheduler):
    remote_ckpt_dir = f"gdrive:Kronos/outputs/{cfg.exp_name}/tokenizer/checkpoints"
    _gdrive_restore_checkpoints(ckpt_dir, remote_ckpt_dir)
    if not list(ckpt_dir.glob("ckpt-*.pt")) and cfg.hf_repo and cfg.hf_checkpoint_revision_out:
        restore_checkpoints(exp_root, cfg.hf_repo, "tokenizer/checkpoints", cfg.hf_checkpoint_revision_out)
    return _load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler)


def _backup_tokenizer_checkpoint(cfg: Config, ckpt_dir: Path, global_step: int) -> None:
    ckpt_path = ckpt_dir / f"ckpt-{global_step}.pt"
    _gdrive_sync_checkpoint(ckpt_path, f"gdrive:Kronos/outputs/{cfg.exp_name}/tokenizer/checkpoints")
    if cfg.hf_repo and cfg.hf_checkpoint_revision_out:
        push_checkpoint(
            ckpt_path,
            cfg.hf_repo,
            f"tokenizer/checkpoints/{ckpt_path.name}",
            cfg.hf_checkpoint_revision_out,
            cfg.hf_checkpoint_keep_last_n,
        )
```

```python
exp_root = Path(cfg.output_dir) / cfg.exp_name
save_dir = exp_root / "tokenizer"
ckpt_dir = save_dir / "checkpoints"

start_epoch, global_step = _restore_tokenizer_training_state(
    cfg,
    exp_root,
    ckpt_dir,
    tokenizer,
    optimizer,
    scheduler,
)

if global_step % cfg.save_steps == 0:
    _save_checkpoint(ckpt_dir, global_step, epoch, tokenizer, optimizer, scheduler)
    _backup_tokenizer_checkpoint(cfg, ckpt_dir, global_step)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/finetune_tw/test_train_tokenizer.py -k "restore_tokenizer_training_state or backup_tokenizer_checkpoint" -v`

Expected: PASS, showing tokenizer now follows the same restore and backup path as predictor.

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/train_tokenizer.py tests/finetune_tw/test_train_tokenizer.py
git commit -m "feat: add tokenizer hf checkpoint fallback"
```

### Task 5: One-Command MoLab Resume Script

**Files:**
- Create: `scripts/resume_molab_training.sh`
- Create: `tests/finetune_tw/test_resume_molab_training_sh.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "resume_molab_training.sh"


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_resume_script_rejects_config_outside_state_dir(tmp_path):
    state_dir = tmp_path / "state"
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "db_path: /marimo/Kronos/finetune_tw/data/tw_stocks.db\n"
        "output_dir: /marimo/Kronos/finetune_tw/outputs\n"
        "exp_name: tw_daily\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "--config",
            str(cfg),
            "--repo-dir",
            str(tmp_path / "repo"),
            "--state-dir",
            str(state_dir),
            "--stage",
            "predictor",
        ],
        env={**os.environ, "KRONOS_SKIP_MONITOR": "1"},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must live under state-dir" in result.stderr


def test_resume_script_reclones_invalid_repo_and_launches_training(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "README.txt").write_text("broken checkout\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    cfg = tmp_path / "good.yaml"
    cfg.write_text(
        f"db_path: {state_dir}/data/tw_stocks.db\n"
        f"output_dir: {state_dir}/outputs\n"
        "exp_name: tw_daily\n",
        encoding="utf-8",
    )

    git_log = tmp_path / "git.jsonl"
    launch_log = tmp_path / "launch.jsonl"
    fake_git = tmp_path / "fake-git"
    fake_launch = tmp_path / "fake-launch"
    _write_exe(
        fake_git,
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['KRONOS_TEST_GIT_LOG'])\n"
        "with log.open('a', encoding='utf-8') as fh:\n"
        "    fh.write(json.dumps({'argv': sys.argv[1:]}) + '\\n')\n"
        "if sys.argv[1] == 'clone':\n"
        "    repo_dir = Path(sys.argv[-1])\n"
        "    repo_dir.mkdir(parents=True, exist_ok=True)\n"
        "    (repo_dir / '.git').mkdir(exist_ok=True)\n"
        "    sys.exit(0)\n"
        "if len(sys.argv) >= 5 and sys.argv[1] == '-C' and sys.argv[3] == 'rev-parse':\n"
        "    repo_dir = Path(sys.argv[2])\n"
        "    sys.exit(0 if (repo_dir / '.git').exists() else 1)\n"
        "sys.exit(0)\n",
    )
    _write_exe(
        fake_launch,
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['KRONOS_TEST_LAUNCH_LOG'])\n"
        "with log.open('a', encoding='utf-8') as fh:\n"
        "    fh.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "KRONOS_GIT_BIN": str(fake_git),
            "KRONOS_LAUNCH_PYTHON": str(fake_launch),
            "KRONOS_MONITOR_ONESHOT": "1",
            "KRONOS_TEST_GIT_LOG": str(git_log),
            "KRONOS_TEST_LAUNCH_LOG": str(launch_log),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "--config",
            str(cfg),
            "--repo-dir",
            str(repo_dir),
            "--state-dir",
            str(state_dir),
            "--stage",
            "predictor",
            "--branch",
            "feature/atr-vol-open-ic",
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    time.sleep(0.2)
    assert (repo_dir / ".git").exists()
    assert (state_dir / "run" / "predictor.pid").exists()
    git_calls = [json.loads(line)["argv"] for line in git_log.read_text(encoding="utf-8").splitlines()]
    assert any(call[0] == "clone" for call in git_calls)
    launch_calls = [json.loads(line) for line in launch_log.read_text(encoding="utf-8").splitlines()]
    assert launch_calls[0]["argv"] == ["-m", "finetune_tw.train_predictor", "--config", str(cfg)]
    assert launch_calls[0]["cwd"] == str(repo_dir)


def test_resume_script_creates_state_skeleton_and_monitor_log(tmp_path):
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    state_dir = tmp_path / "state"
    cfg = tmp_path / "good.yaml"
    cfg.write_text(
        f"db_path: {state_dir}/data/tw_stocks.db\n"
        f"output_dir: {state_dir}/outputs\n"
        "exp_name: tw_daily\n",
        encoding="utf-8",
    )
    fake_git = tmp_path / "fake-git"
    fake_launch = tmp_path / "fake-launch"
    _write_exe(fake_git, "#!/usr/bin/env bash\nexit 0\n")
    _write_exe(fake_launch, "#!/usr/bin/env bash\nexit 0\n")

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "--config",
            str(cfg),
            "--repo-dir",
            str(repo_dir),
            "--state-dir",
            str(state_dir),
            "--stage",
            "tokenizer",
        ],
        env={
            **os.environ,
            "KRONOS_GIT_BIN": str(fake_git),
            "KRONOS_LAUNCH_PYTHON": str(fake_launch),
            "KRONOS_MONITOR_ONESHOT": "1",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    time.sleep(0.2)
    assert (state_dir / "data").exists()
    assert (state_dir / "outputs").exists()
    assert (state_dir / "logs" / "tokenizer_monitor.log").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/finetune_tw/test_resume_molab_training_sh.py -v`

Expected: FAIL because `scripts/resume_molab_training.sh` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_url="https://github.com/j835111/Kronos.git"
repo_dir="/marimo/Kronos"
state_dir="/mnt/first/kronos_state"
stage="predictor"
branch=""
config=""
git_bin="${KRONOS_GIT_BIN:-git}"
launch_python="${KRONOS_LAUNCH_PYTHON:-python3}"
monitor_interval="${KRONOS_MONITOR_INTERVAL_SECONDS:-600}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="$2"; shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    --repo-url) repo_url="$2"; shift 2 ;;
    --repo-dir) repo_dir="$2"; shift 2 ;;
    --state-dir) state_dir="$2"; shift 2 ;;
    --branch) branch="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$config" ]] || { echo "--config is required" >&2; exit 2; }
[[ "$stage" == "tokenizer" || "$stage" == "predictor" ]] || { echo "--stage must be tokenizer or predictor" >&2; exit 2; }

read_cfg() {
  python3 - "$config" "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg.get(sys.argv[2], ""))
PY
}

ensure_state_skeleton() {
  mkdir -p "$state_dir/data" "$state_dir/outputs" "$state_dir/logs" "$state_dir/run"
}

ensure_repo_checkout() {
  if ! "$git_bin" -C "$repo_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    rm -rf "$repo_dir"
    "$git_bin" clone "$repo_url" "$repo_dir"
  fi
  "$git_bin" -C "$repo_dir" fetch --all --prune || true
  if [[ -n "$branch" ]]; then
    "$git_bin" -C "$repo_dir" checkout "$branch"
    "$git_bin" -C "$repo_dir" reset --hard "origin/$branch" || true
  fi
}

stop_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    kill "$(cat "$pid_file")" 2>/dev/null || true
    rm -f "$pid_file"
  fi
}

start_monitor() {
  local exp_name="$1"
  local monitor_log="$state_dir/logs/${stage}_monitor.log"
  local monitor_pid="$state_dir/run/${stage}_monitor.pid"
  stop_pid_file "$monitor_pid"
  (
    while true; do
      latest_ckpt="$(find "$state_dir/outputs/$exp_name/$stage/checkpoints" -maxdepth 1 -name 'ckpt-*.pt' 2>/dev/null | sort -V | tail -n1 || true)"
      latest_log="$(tail -n1 "$state_dir/outputs/$exp_name/$stage/train_log.csv" 2>/dev/null || true)"
      printf '%s stage=%s ckpt=%s log=%s\n' "$(date -Iseconds)" "$stage" "${latest_ckpt##*/}" "$latest_log" >> "$monitor_log"
      if [[ "${KRONOS_MONITOR_ONESHOT:-0}" == "1" ]]; then
        exit 0
      fi
      sleep "$monitor_interval"
    done
  ) &
  echo $! > "$monitor_pid"
}

launch_training() {
  local module="$1"
  local stdout_log="$state_dir/logs/${stage}_train_stdout.log"
  local train_pid="$state_dir/run/${stage}.pid"
  stop_pid_file "$train_pid"
  (
    cd "$repo_dir"
    exec "$launch_python" -m "$module" --config "$config"
  ) >> "$stdout_log" 2>&1 &
  echo $! > "$train_pid"
}

ensure_state_skeleton
db_path="$(read_cfg db_path)"
output_dir="$(read_cfg output_dir)"
exp_name="$(read_cfg exp_name)"

case "$db_path" in
  "$state_dir"/*) ;;
  *) echo "db_path must live under state-dir: $state_dir" >&2; exit 1 ;;
esac

case "$output_dir" in
  "$state_dir"/*) ;;
  *) echo "output_dir must live under state-dir: $state_dir" >&2; exit 1 ;;
esac

ensure_repo_checkout

if [[ "$stage" == "tokenizer" ]]; then
  launch_training "finetune_tw.train_tokenizer"
else
  launch_training "finetune_tw.train_predictor"
fi

if [[ "${KRONOS_SKIP_MONITOR:-0}" != "1" ]]; then
  start_monitor "$exp_name"
fi
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/finetune_tw/test_resume_molab_training_sh.py -v`

Expected: PASS for config-path rejection, invalid-repo re-clone, launch, state skeleton creation, and monitor log creation.

- [ ] **Step 5: Commit**

```bash
git add scripts/resume_molab_training.sh tests/finetune_tw/test_resume_molab_training_sh.py
git commit -m "feat: add molab resume bootstrap script"
```

### Task 6: Full Regression Sweep For The Backup Path

**Files:**
- Modify: `finetune_tw/hf_utils.py`
- Modify: `finetune_tw/train_predictor.py`
- Modify: `finetune_tw/train_tokenizer.py`
- Modify: `scripts/resume_molab_training.sh`
- Test: `tests/finetune_tw/test_hf_utils.py`
- Test: `tests/finetune_tw/test_train_predictor.py`
- Test: `tests/finetune_tw/test_train_tokenizer.py`
- Test: `tests/finetune_tw/test_resume_molab_training_sh.py`
- Test: `tests/finetune_tw/test_config_retrain.py`

- [ ] **Step 1: Run the focused regression suite**

Run: `pytest tests/finetune_tw/test_config_retrain.py tests/finetune_tw/test_hf_utils.py tests/finetune_tw/test_train_predictor.py tests/finetune_tw/test_train_tokenizer.py tests/finetune_tw/test_resume_molab_training_sh.py -v`

Expected: PASS with new restore, backup, prune, and resume-script behavior covered.

- [ ] **Step 2: Run a wider finetune test sweep**

Run: `pytest tests/finetune_tw -v`

Expected: PASS with no regressions in existing predictor, tokenizer, or script tests.

- [ ] **Step 3: Dry-run the new resume script locally with monitor disabled**

Run: `bash scripts/resume_molab_training.sh --config finetune_tw/configs/config_tw_daily_rtx6000.yaml --stage predictor --repo-dir /tmp/kronos-repo --state-dir /tmp/kronos-state --branch feature/atr-vol-open-ic`

Expected: Exit early unless `/tmp/kronos-state` config paths match the temp state dir. This is intentional; the script must reject dangerous configs that still point outside `state-dir`.

- [ ] **Step 4: Perform the real MoLab smoke test after code review**

Run: `bash scripts/resume_molab_training.sh --config finetune_tw/configs/config_tw_daily_rtx6000.yaml --stage predictor --repo-dir /marimo/Kronos --state-dir /mnt/first/kronos_state --branch feature/atr-vol-open-ic`

Expected: With existing local checkpoints under `/mnt/first/kronos_state/outputs/tw_daily/predictor/checkpoints`, the script resumes from the latest local `ckpt-*.pt`, writes `predictor.pid` and `predictor_train_stdout.log`, and the monitor writes `predictor_monitor.log`.

- [ ] **Step 5: Perform the HF fallback recovery drill**

Run: `mv /mnt/first/kronos_state/outputs/tw_daily/predictor/checkpoints /mnt/first/kronos_state/outputs/tw_daily/predictor/checkpoints.bak && bash scripts/resume_molab_training.sh --config finetune_tw/configs/config_tw_daily_rtx6000.yaml --stage predictor --repo-dir /marimo/Kronos --state-dir /mnt/first/kronos_state --branch feature/atr-vol-open-ic`

Expected: Startup logs show no local predictor checkpoint, HF restore repopulates `predictor/checkpoints`, and the training process resumes from the latest checkpoint downloaded from `checkpoints-round-3`.

- [ ] **Step 6: Perform the disposable-checkout rebuild drill**

Run: `rm -rf /marimo/Kronos && bash scripts/resume_molab_training.sh --config finetune_tw/configs/config_tw_daily_rtx6000.yaml --stage predictor --repo-dir /marimo/Kronos --state-dir /mnt/first/kronos_state --branch feature/atr-vol-open-ic`

Expected: The script re-clones the repo, checks out `feature/atr-vol-open-ic`, leaves `/mnt/first/kronos_state` untouched, resumes training, and recreates both training and monitor pid files under `state-dir/run`.

- [ ] **Step 7: Commit the final integration batch**

```bash
git add finetune_tw/hf_utils.py finetune_tw/train_predictor.py finetune_tw/train_tokenizer.py scripts/resume_molab_training.sh tests/finetune_tw/test_config_retrain.py tests/finetune_tw/test_hf_utils.py tests/finetune_tw/test_train_predictor.py tests/finetune_tw/test_train_tokenizer.py tests/finetune_tw/test_resume_molab_training_sh.py
git commit -m "feat: add persistent molab checkpoint recovery"
```
