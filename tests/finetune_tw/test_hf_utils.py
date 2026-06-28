import sys
import threading
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

    def create_branch(
        self,
        repo_id: str,
        *,
        branch: str,
        token: str | None = None,
        exist_ok: bool = False,
        revision: str | None = None,
        repo_type: str | None = None,
    ) -> None:
        self.branches.append((repo_id, branch, exist_ok))

    def list_repo_tree(
        self,
        repo_id: str,
        path_in_repo: str | None = None,
        *,
        recursive: bool = False,
        expand: bool = False,
        revision: str | None = None,
        repo_type: str | None = None,
        token: str | None = None,
    ):
        return [FakeRepoFile(path) for path in self.tree]

    def delete_file(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        revision: str | None = None,
        token: str | None = None,
        repo_type: str | None = None,
        commit_message: str | None = None,
        commit_description: str | None = None,
        create_pr: bool | None = None,
        parent_commit: str | None = None,
    ) -> None:
        self.deleted.append((path_in_repo, revision))


def test_local_checkpoints_filters_invalid_entries(tmp_path):
    ckpt_dir = tmp_path / "predictor" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-100.pt").write_bytes(b"x")
    (ckpt_dir / "ckpt-200.pt").write_bytes(b"")
    (ckpt_dir / "ckpt-latest.pt").write_bytes(b"x")
    (ckpt_dir / "ckpt-300.pt").mkdir()

    checkpoints = hf_utils.local_checkpoints(ckpt_dir)

    assert checkpoints == [ckpt_dir / "ckpt-100.pt"]


def test_restore_checkpoints_skips_remote_when_local_exists(tmp_path, monkeypatch):
    exp_dir = tmp_path / "tw_daily"
    ckpt_dir = exp_dir / "predictor" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-500.pt").write_bytes(b"x")
    called: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **kwargs: called.setdefault("download", kwargs)),
    )

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

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    )

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

    hf_utils.prune_checkpoints("repo", "predictor/checkpoints", "checkpoints-round-3", 3, api=api)

    assert api.deleted == [
        ("predictor/checkpoints/ckpt-100.pt", "checkpoints-round-3"),
        ("predictor/checkpoints/ckpt-200.pt", "checkpoints-round-3"),
    ]


def test_prune_checkpoints_ignores_malformed_remote_names(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    api = FakeHfApi(
        tree=[
            "predictor/checkpoints/ckpt-100.pt",
            "predictor/checkpoints/ckpt-latest.pt",
            "predictor/checkpoints/ckpt-200.pt",
            "predictor/checkpoints/notes.pt",
        ]
    )

    hf_utils.prune_checkpoints("repo", "predictor/checkpoints", "checkpoints-round-3", 1, api=api)

    assert api.deleted == [("predictor/checkpoints/ckpt-100.pt", "checkpoints-round-3")]


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


def test_push_checkpoint_missing_file_is_swallowed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "token")
    missing = tmp_path / "ckpt-999.pt"
    fake_api = FakeHfApi()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: fake_api, upload_file=lambda **kwargs: None),
    )

    hf_utils.push_checkpoint(
        missing,
        "repo",
        "predictor/checkpoints/ckpt-999.pt",
        "checkpoints-round-3",
        3,
    )
    hf_utils.wait_for_pushes()

    captured = capsys.readouterr()
    assert "checkpoint push failed" in captured.out


def test_push_checkpoint_creates_branch_uploads_and_prunes(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    ckpt = tmp_path / "ckpt-600.pt"
    ckpt.write_bytes(b"payload")
    fake_api = FakeHfApi()
    calls: list[tuple[str, object]] = []

    def fake_upload_file(**kwargs):
        assert kwargs["path_or_fileobj"].read() == b"payload"
        calls.append(("upload", kwargs))

    def fake_prune_checkpoints(repo_id, subfolder, revision, keep_last_n, api=None):
        calls.append(("prune", (repo_id, subfolder, revision, keep_last_n, api)))

    monkeypatch.setattr(hf_utils, "prune_checkpoints", fake_prune_checkpoints)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: fake_api, upload_file=fake_upload_file),
    )

    hf_utils.push_checkpoint(
        ckpt,
        "repo",
        "predictor/checkpoints/ckpt-600.pt",
        "checkpoints-round-3",
        3,
    )
    hf_utils.wait_for_pushes()

    assert fake_api.branches == [("repo", "checkpoints-round-3", True)]
    assert calls[0][0] == "upload"
    assert calls[0][1]["path_in_repo"] == "predictor/checkpoints/ckpt-600.pt"
    assert calls[1] == (
        "prune",
        ("repo", "predictor/checkpoints", "checkpoints-round-3", 3, fake_api),
    )


def test_push_checkpoint_keeps_open_file_across_local_unlink(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    ckpt = tmp_path / "ckpt-700.pt"
    ckpt.write_bytes(b"payload")
    fake_api = FakeHfApi()
    uploaded: dict[str, bytes] = {}

    class DelayedThread:
        def __init__(self, target, daemon=False, name=None):
            self._target = target
            self._alive = False

        def start(self):
            ckpt.unlink()
            self._alive = True
            try:
                self._target()
            finally:
                self._alive = False

        def is_alive(self):
            return self._alive

        def join(self):
            return None

    def fake_upload_file(**kwargs):
        uploaded["payload"] = kwargs["path_or_fileobj"].read()

    monkeypatch.setattr(hf_utils.threading, "Thread", DelayedThread)
    monkeypatch.setattr(hf_utils, "prune_checkpoints", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: fake_api, upload_file=fake_upload_file),
    )

    hf_utils.push_checkpoint(
        ckpt,
        "repo",
        "predictor/checkpoints/ckpt-700.pt",
        "checkpoints-round-3",
        3,
    )

    assert uploaded["payload"] == b"payload"


def test_wait_for_pushes_prunes_completed_threads():
    done = threading.Thread(target=lambda: None)
    done.start()
    done.join()
    hf_utils._push_threads[:] = [done]

    hf_utils.wait_for_pushes()

    assert hf_utils._push_threads == []
