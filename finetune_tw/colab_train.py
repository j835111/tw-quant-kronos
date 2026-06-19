"""
Self-contained Kronos fine-tuning runner for Google Colab CLI.

Usage:
    # 1. Provision session
    colab new -s kronos-trainer --gpu T4

    # 2. (Optional) Upload pre-built DB to skip download (~5 min saved)
    #    colab upload -s kronos-trainer /home/james/kronos_data/tw_stocks.db /content/tw_stocks.db

    # 3. (Optional) Upload local finetune_tw patch (if fork not updated)
    #    colab upload -s kronos-trainer /tmp/finetune_tw.tar.gz /content/finetune_tw.tar.gz

    # 4. Run
    colab exec -s kronos-trainer -f finetune_tw/colab_train.py --timeout 10800
"""
import os, sys, subprocess, dataclasses, tarfile
from pathlib import Path

REPO_URL  = "https://github.com/j835111/Kronos.git"
REPO_DIR  = "/content/Kronos"
DB_UPLOAD = "/content/tw_stocks.db"          # pre-uploaded (optional)
DB_DST    = "/content/Kronos/finetune_tw/data/tw_stocks.db"
FTW_TAR   = "/content/finetune_tw.tar.gz"    # pre-uploaded patch (optional)

# ── 0. rclone setup ──────────────────────────────────────────────────────────
_RCLONE_CONF_SRC = "/content/rclone.conf"   # upload via: colab upload ... /content/rclone.conf
_RCLONE_CONF_DST = Path.home() / ".config/rclone/rclone.conf"
if Path(_RCLONE_CONF_SRC).exists() and not _RCLONE_CONF_DST.exists():
    import shutil as _sh
    _RCLONE_CONF_DST.parent.mkdir(parents=True, exist_ok=True)
    _sh.copy2(_RCLONE_CONF_SRC, _RCLONE_CONF_DST)
    _RCLONE_CONF_DST.chmod(0o600)
    print(f"rclone.conf installed → {_RCLONE_CONF_DST}")

# Install rclone if needed
import shutil as _sh2
if _sh2.which("rclone") is None:
    subprocess.run(["apt-get", "install", "-y", "-q", "rclone"], check=False)

# ── 1. Clone repo ─────────────────────────────────────────────────────────────
if not Path(REPO_DIR).exists():
    print("Cloning repo...")
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, REPO_DIR], check=True)
else:
    print("Repo already cloned.")

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# ── 2. Patch finetune_tw (from tar.gz if available) ──────────────────────────
if Path(FTW_TAR).exists():
    print(f"Extracting {FTW_TAR} over repo...")
    with tarfile.open(FTW_TAR, "r:gz") as tar:
        tar.extractall(REPO_DIR)
    print("Extracted.")
elif not Path("finetune_tw").exists():
    raise FileNotFoundError(
        "finetune_tw/ not found and no patch tar.gz at /content/finetune_tw.tar.gz. "
        "Either push finetune_tw to the fork, or upload the tar.gz:\n"
        "  cd /mnt/d/project/Kronos && "
        "  tar -czf /tmp/finetune_tw.tar.gz --exclude=finetune_tw/data "
        "--exclude=finetune_tw/outputs --exclude='*/__pycache__' finetune_tw/\n"
        "  colab upload -s kronos-trainer /tmp/finetune_tw.tar.gz /content/finetune_tw.tar.gz"
    )
else:
    print("finetune_tw/ already present (from fork).")

# ── 3. Install dependencies ───────────────────────────────────────────────────
print("\nInstalling dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "yfinance", "pyyaml", "tqdm"],
    check=True,
)

# ── 4. Place / download DB ────────────────────────────────────────────────────
Path(DB_DST).parent.mkdir(parents=True, exist_ok=True)
if not Path(DB_DST).exists():
    if Path(DB_UPLOAD).exists():
        import shutil
        shutil.copy2(DB_UPLOAD, DB_DST)
        print(f"DB copied: {DB_UPLOAD} → {DB_DST}")
    else:
        print("DB not found — downloading Taiwan stock data on Colab VM...")
        print("(This takes ~3–5 minutes on Colab's fast network)")
        from finetune_tw.download_data import download as _dl
        from finetune_tw.fetchers.yfinance_fetcher import get_twse_symbol_list as _syms
        _symbols = ["^TWII"] + _syms()
        _dl(db_path=DB_DST, symbols=_symbols, start="2015-01-01",
            end=str(__import__("datetime").date.today()), source="yfinance")
        print("Download complete.")
else:
    print(f"DB already at {DB_DST}")

# ── 5. Resume detection (don't wipe if Drive has checkpoints) ────────────────
_stale = Path("finetune_tw/outputs")
_has_remote = False
if _sh2.which("rclone"):
    _r = subprocess.run(
        ["rclone", "lsf", f"gdrive:Kronos/outputs/tw_daily/tokenizer/checkpoints/"],
        capture_output=True, text=True, timeout=30,
    )
    _has_remote = bool(_r.stdout.strip())

if not _has_remote and _stale.exists():
    _stale_ckpts = list(_stale.rglob("ckpt-*.pt"))
    if _stale_ckpts:
        print(f"No Drive checkpoints found. Removing {len(_stale_ckpts)} stale local checkpoint(s)...")
        import shutil as _sh3
        _sh3.rmtree(str(_stale))
        _stale.mkdir(parents=True, exist_ok=True)
elif _has_remote:
    print("Drive checkpoints found — will resume from Drive.")

# ── 6. Build config ───────────────────────────────────────────────────────────
import torch
from finetune_tw.config import Config

cfg = Config.from_yaml("finetune_tw/configs/config_tw_daily.yaml")
cfg = dataclasses.replace(cfg,
    db_path=DB_DST,
    tokenizer_epochs=2,
    basemodel_epochs=1,
    batch_size=64,
    num_workers=4,
    save_steps=200,
    log_interval=20,
)

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU!)"
print(f"\nDevice : {device_name}")
print(f"Config : tokenizer_epochs={cfg.tokenizer_epochs}, "
      f"basemodel_epochs={cfg.basemodel_epochs}, "
      f"batch_size={cfg.batch_size}")

# ── 7. Train tokenizer ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 1 / 3 — Tokenizer Training  [fp32, no autocast]")
print("="*60)
from finetune_tw.train_tokenizer import run_training as train_tok
train_tok(cfg)

# ── 8. Train predictor ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 2 / 3 — Predictor Training  [fp32, no autocast]")
print("="*60)
from finetune_tw.train_predictor import run_training as train_pred
train_pred(cfg)

# ── 9. Backtest ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 3 / 3 — Backtest (2024-07-01 ~)")
print("="*60)

import yaml
bt_cfg = {
    "db_path": DB_DST,
    "lookback_window": cfg.lookback_window,
    "predict_window": cfg.predict_window,
    "max_context": cfg.max_context,
    "clip": cfg.clip,
    "train_end_date": cfg.train_end_date,
    "val_end_date": cfg.val_end_date,
    "pretrained_tokenizer": cfg.pretrained_tokenizer,
    "pretrained_predictor": cfg.pretrained_predictor,
    "exp_name": cfg.exp_name,
    "output_dir": cfg.output_dir,
    "top_k": cfg.top_k,
    "hold_days": cfg.hold_days,
    "pred_len": cfg.pred_len,
    "test_start_date": cfg.test_start_date,
    "benchmark_symbol": cfg.benchmark_symbol,
}
bt_cfg_path = "/tmp/config_colab_backtest.yaml"
with open(bt_cfg_path, "w") as f:
    yaml.dump(bt_cfg, f)

ret = subprocess.run(
    [sys.executable, "finetune_tw/backtest.py", "--config", bt_cfg_path],
    capture_output=False,
)
if ret.returncode != 0:
    print("Backtest exited with error (non-fatal).")

print("\n" + "="*60)
print(f"All done! Outputs: {cfg.output_dir}/{cfg.exp_name}/")
print("="*60)
