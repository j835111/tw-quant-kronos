"""HuggingFace Hub helpers — push and restore model weights."""
from __future__ import annotations
import os
import threading
from pathlib import Path


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
    _push_threads.append(t)
    t.start()
    print(f"  [hf] push started (background) → {repo_id}@{revision}/{subfolder}")


def wait_for_pushes() -> None:
    """Block until all pending HF push threads complete. Call before process exit."""
    pending = [t for t in _push_threads if t.is_alive()]
    if pending:
        print(f"  [hf] waiting for {len(pending)} upload(s) to complete …")
        for t in pending:
            t.join()
        print("  [hf] all uploads done.")


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
