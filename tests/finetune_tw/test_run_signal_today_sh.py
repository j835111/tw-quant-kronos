import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "run_signal_today.sh"


def test_run_signal_today_script_updates_db_then_runs_signal(tmp_path):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "with open(os.environ['KRONOS_TEST_LOG'], 'a', encoding='utf-8') as fh:\n"
        "    fh.write(json.dumps({\n"
        "        'argv': sys.argv[1:],\n"
        "        'cwd': os.getcwd(),\n"
        "        'mplconfigdir': os.environ.get('MPLCONFIGDIR'),\n"
        "        'tmpdir': os.environ.get('TMPDIR'),\n"
        "    }) + '\\n')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    external_cwd = tmp_path / "outside"
    external_cwd.mkdir()
    log_path = tmp_path / "calls.jsonl"

    env = os.environ.copy()
    env["KRONOS_PYTHON"] = str(fake_python)
    env["KRONOS_TEST_LOG"] = str(log_path)

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=external_cwd,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    calls = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert calls == [
        {
            "argv": [
                "-m",
                "finetune_tw.download_data",
                "--config",
                "finetune_tw/configs/config_tw_daily.yaml",
                "--update",
            ],
            "cwd": str(REPO_ROOT),
            "mplconfigdir": str(REPO_ROOT / ".cache" / "matplotlib"),
            "tmpdir": str(REPO_ROOT / ".tmp"),
        },
        {
            "argv": [
                "-m",
                "finetune_tw.signal_today",
                "--config",
                "finetune_tw/configs/config_tw_daily.yaml",
                "--model",
                "round0",
                "--top_k",
                "10",
                "--hold_days",
                "3",
            ],
            "cwd": str(REPO_ROOT),
            "mplconfigdir": str(REPO_ROOT / ".cache" / "matplotlib"),
            "tmpdir": str(REPO_ROOT / ".tmp"),
        },
    ]
