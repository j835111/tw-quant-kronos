"""HuggingFace Hub helpers — push and restore model weights."""
from __future__ import annotations
import os
import threading
from pathlib import Path, PurePosixPath


def _token() -> str | None:
    return os.environ.get("HF_TOKEN")


def has_weights(local_dir: Path) -> bool:
    """True if local_dir contains model.safetensors (non-empty)."""
    f = local_dir / "model.safetensors"
    return f.exists() and f.stat().st_size > 0


def resolve_src(local_dir: Path, repo_id: str, subfolder: str, revision: str) -> tuple[str, dict]:
    """Return (src_path, from_pretrained_kwargs) — local if available, else HF Hub."""
    if has_weights(local_dir):
        return str(local_dir), {}
    print(f"  [hf] local weights missing at {local_dir}, loading from {repo_id}@{revision}/{subfolder}")
    return repo_id, {"subfolder": subfolder, "revision": revision, "token": _token()}


_push_threads: list[threading.Thread] = []


def _checkpoint_step(path: Path | str) -> int:
    name = Path(path).name
    return int(name.removeprefix("ckpt-").removesuffix(".pt"))


def local_checkpoints(ckpt_dir: Path) -> list[Path]:
    checkpoints: list[Path] = []
    for path in ckpt_dir.glob("ckpt-*.pt"):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        try:
            _checkpoint_step(path)
        except ValueError:
            continue
        checkpoints.append(path)
    return sorted(checkpoints, key=_checkpoint_step)


def _checkpoint_repo_paths(items) -> list[str]:
    checkpoints: list[str] = []
    for item in items:
        path = item.path
        name = PurePosixPath(path).name
        if not name.startswith("ckpt-") or not path.endswith(".pt"):
            continue
        try:
            _checkpoint_step(path)
        except ValueError:
            continue
        checkpoints.append(path)
    return sorted(checkpoints, key=_checkpoint_step)


def _track_push_thread(thread: threading.Thread) -> None:
    _push_threads[:] = [t for t in _push_threads if t.is_alive()]
    _push_threads.append(thread)


def push_best_model(local_dir: Path, repo_id: str, subfolder: str, revision: str) -> None:
    """Upload local_dir to HF Hub in a background thread. Skips if HF_TOKEN not set."""
    token = _token()
    if not token:
        print(f"  [hf] HF_TOKEN not set — skipping push to {repo_id}@{revision}/{subfolder}")
        return

    def _run() -> None:
        try:
            from huggingface_hub import upload_folder
            commit = upload_folder(
                repo_id=repo_id,
                folder_path=str(local_dir),
                path_in_repo=subfolder,
                revision=revision,
                token=token,
                commit_message=f"update {subfolder}",
            )
            print(f"  [hf] pushed {subfolder} → {commit}")
        except Exception as exc:
            print(f"  [hf] push failed: {exc}")

    t = threading.Thread(target=_run, daemon=False, name="hf-push")
    _track_push_thread(t)
    t.start()
    print(f"  [hf] push started (background) → {repo_id}@{revision}/{subfolder}")


def push_file(local_path: Path, repo_id: str, path_in_repo: str, revision: str) -> None:
    """Upload a single file to HF Hub in a background thread."""
    token = _token()
    if not token:
        return

    def _run() -> None:
        try:
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                revision=revision,
                token=token,
                commit_message=f"update {path_in_repo}",
            )
        except Exception as exc:
            print(f"  [hf] push_file failed: {exc}")

    t = threading.Thread(target=_run, daemon=False, name="hf-push-file")
    _track_push_thread(t)
    t.start()


def wait_for_pushes() -> None:
    """Block until all pending HF push threads complete. Call before process exit."""
    pending = [t for t in _push_threads if t.is_alive()]
    if pending:
        print(f"  [hf] waiting for {len(pending)} upload(s) to complete …")
        for t in pending:
            t.join()
        print("  [hf] all uploads done.")
    _push_threads[:] = [t for t in _push_threads if t.is_alive()]


def restore_checkpoints(exp_dir: Path, repo_id: str, subfolder: str, revision: str) -> int:
    """Download rolling checkpoints into exp_dir/subfolder if none exist locally."""
    ckpt_dir = exp_dir / subfolder
    if local_checkpoints(ckpt_dir):
        return 0
    token = _token()
    print(f"  [hf] restoring {subfolder} from {repo_id}@{revision} …")
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
            print(f"  [hf] restore: no checkpoints found at {repo_id}@{revision}/{subfolder}")
        return restored
    except Exception as exc:
        print(f"  [hf] checkpoint restore failed: {exc}")
        return 0


def prune_checkpoints(
    repo_id: str,
    subfolder: str,
    revision: str,
    keep_last_n: int,
    api=None,
) -> None:
    """Delete older remote checkpoints, keeping the most recent N by step number."""
    token = _token()
    if not token:
        return
    try:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        files = _checkpoint_repo_paths(
            api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=subfolder,
                revision=revision,
                token=token,
                recursive=False,
            )
        )
        stale = files[:-keep_last_n] if keep_last_n > 0 else files
    except Exception as exc:
        print(f"  [hf] checkpoint prune failed: {exc}")
        return

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


def push_checkpoint(
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    revision: str,
    keep_last_n: int,
) -> None:
    """Upload a checkpoint in a background thread, then prune older remote checkpoints."""
    token = _token()
    if not token:
        print(f"  [hf] HF_TOKEN not set — skipping checkpoint push to {repo_id}@{revision}/{path_in_repo}")
        return
    try:
        fileobj = local_path.open("rb")
    except Exception as exc:
        print(f"  [hf] checkpoint push failed: {exc}")
        return

    def _run() -> None:
        try:
            from huggingface_hub import HfApi, upload_file
            api = HfApi()
            api.create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
            upload_file(
                path_or_fileobj=fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                revision=revision,
                token=token,
                commit_message=f"update {path_in_repo}",
            )
            prune_checkpoints(
                repo_id=repo_id,
                subfolder=str(PurePosixPath(path_in_repo).parent),
                revision=revision,
                keep_last_n=keep_last_n,
                api=api,
            )
        except Exception as exc:
            print(f"  [hf] checkpoint push failed: {exc}")
        finally:
            fileobj.close()

    t = threading.Thread(target=_run, daemon=False, name="hf-push-checkpoint")
    _track_push_thread(t)
    t.start()


def restore_best_model(exp_dir: Path, repo_id: str, subfolder: str, revision: str) -> bool:
    """Download subfolder from HF into exp_dir if local weights are missing. Returns True if fetched."""
    local_dir = exp_dir / subfolder
    if has_weights(local_dir):
        return False
    token = _token()
    print(f"  [hf] restoring {subfolder} from {repo_id}@{revision} …")
    try:
        from huggingface_hub import snapshot_download
        exp_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[f"{subfolder}/*"],
            local_dir=str(exp_dir),
            token=token,
        )
        if has_weights(local_dir):
            print(f"  [hf] restored {subfolder} ✓")
            return True
        print(f"  [hf] restore: {subfolder}/model.safetensors not found in repo")
        return False
    except Exception as exc:
        print(f"  [hf] restore failed: {exc}")
        return False
