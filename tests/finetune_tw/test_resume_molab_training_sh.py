import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "resume_molab_training.sh"


def _write_config(path: Path, state_dir: Path) -> None:
    output_dir = state_dir / "outputs"
    config_text = textwrap.dedent(
        f"""\
        db_path: "{state_dir / 'data' / 'tw_stocks.db'}"
        output_dir: "{output_dir}"
        exp_name: "demo_exp"
        """
    )
    path.write_text(config_text, encoding="utf-8")


def _write_fake_git(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from pathlib import Path

            log_path = Path(os.environ["KRONOS_TEST_GIT_LOG"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")

            args = sys.argv[1:]
            if len(args) >= 4 and args[0] == "-C" and args[2:4] == ["rev-parse", "--is-inside-work-tree"]:
                repo_dir = Path(args[1])
                if (repo_dir / ".git-valid").exists():
                    print("true")
                    raise SystemExit(0)
                raise SystemExit(128)

            if len(args) >= 3 and args[0] == "clone":
                repo_dir = Path(args[2])
                repo_dir.mkdir(parents=True, exist_ok=True)
                (repo_dir / ".git-valid").write_text("ok", encoding="utf-8")
                raise SystemExit(0)

            raise SystemExit(0)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_fake_python(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import sys
            import time
            from pathlib import Path

            def read_scalar(config_path: Path, key: str) -> str:
                for raw_line in config_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" not in line:
                        continue
                    current_key, value = line.split(":", 1)
                    if current_key.strip() != key:
                        continue
                    value = value.strip()
                    if value.startswith(("\\"", "'")) and value.endswith(("\\"", "'")):
                        value = value[1:-1]
                    return value
                raise KeyError(key)

            log_path = Path(os.environ["KRONOS_TEST_PYTHON_LOG"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "argv": sys.argv[1:],
                "cwd": os.getcwd(),
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\\n")

            config_path = Path(sys.argv[sys.argv.index("--config") + 1])
            output_dir = Path(read_scalar(config_path, "output_dir"))
            exp_name = read_scalar(config_path, "exp_name")
            stage = "tokenizer" if "train_tokenizer" in sys.argv else "predictor"
            stage_dir = output_dir / exp_name / stage
            ckpt_dir = stage_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "ckpt-0001.pt").write_text("checkpoint", encoding="utf-8")
            (stage_dir / "train_log.csv").write_text(
                "epoch,step,train_loss,val_loss\\n1,10,0.1,0.2\\n",
                encoding="utf-8",
            )
            time.sleep(0.2)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _base_env(tmp_path: Path, git_bin: Path, python_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["KRONOS_GIT_BIN"] = str(git_bin)
    env["KRONOS_LAUNCH_PYTHON"] = str(python_bin)
    env["KRONOS_TEST_GIT_LOG"] = str(tmp_path / "git.jsonl")
    env["KRONOS_TEST_PYTHON_LOG"] = str(tmp_path / "python.jsonl")
    env["KRONOS_MONITOR_ONESHOT"] = "1"
    return env


@pytest.mark.parametrize(
    ("db_path", "output_dir", "expected_error"),
    [
        ("outside.db", "state/outputs", "db_path must live under state-dir"),
        ("state/data/tw_stocks.db", "outside_outputs", "output_dir must live under state-dir"),
    ],
)
def test_resume_script_rejects_paths_outside_state_dir(
    tmp_path, db_path, output_dir, expected_error
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""\
            db_path: "{tmp_path / db_path}"
            output_dir: "{tmp_path / output_dir}"
            exp_name: "demo_exp"
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--config", str(config_path), "--state-dir", str(state_dir)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_resume_script_reclones_invalid_repo_and_launches_training(tmp_path):
    state_dir = tmp_path / "state"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "stale.txt").write_text("stale", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, state_dir)

    fake_git = tmp_path / "fake-git"
    fake_python = tmp_path / "fake-python"
    _write_fake_git(fake_git)
    _write_fake_python(fake_python)
    env = _base_env(tmp_path, fake_git, fake_python)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "--config",
            str(config_path),
            "--repo-url",
            "https://example.com/repo.git",
            "--repo-dir",
            str(repo_dir),
            "--state-dir",
            str(state_dir),
            "--branch",
            "main",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    git_calls = [
        json.loads(line)
        for line in (tmp_path / "git.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert git_calls[0]["argv"] == [
        "-C",
        str(repo_dir),
        "rev-parse",
        "--is-inside-work-tree",
    ]
    assert any(call["argv"][:3] == ["clone", "https://example.com/repo.git", str(repo_dir)] for call in git_calls)
    assert any(call["argv"][:4] == ["-C", str(repo_dir), "fetch", "origin"] for call in git_calls)
    assert not (repo_dir / "stale.txt").exists()

    launches = [
        json.loads(line)
        for line in (tmp_path / "python.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert launches == [
        {
            "argv": [
                "-m",
                "finetune_tw.train_predictor",
                "--config",
                str(config_path),
            ],
            "cwd": str(repo_dir),
        }
    ]


def test_resume_script_creates_state_skeleton_and_monitor_log(tmp_path):
    state_dir = tmp_path / "state"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git-valid").write_text("ok", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, state_dir)

    fake_git = tmp_path / "fake-git"
    fake_python = tmp_path / "fake-python"
    _write_fake_git(fake_git)
    _write_fake_python(fake_python)
    env = _base_env(tmp_path, fake_git, fake_python)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "--config",
            str(config_path),
            "--repo-dir",
            str(repo_dir),
            "--state-dir",
            str(state_dir),
            "--stage",
            "tokenizer",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for name in ("data", "outputs", "logs", "run"):
        assert (state_dir / name).is_dir()
    assert (state_dir / "logs" / "tokenizer_monitor.log").exists()
    assert (state_dir / "run" / "tokenizer.pid").exists()
    assert (state_dir / "run" / "tokenizer_monitor.pid").exists()
