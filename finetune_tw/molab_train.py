"""
Kronos Taiwan Stock Fine-tuning — molab edition
================================================
在 molab.marimo.io 開啟此檔案：
  1. 前往 https://molab.marimo.io
  2. 新建 notebook，點右上角 notebook specs → 啟用 GPU
  3. 執行所有 cell（Run All）

所有路徑在 /marimo/Kronos/ 下（與 repo 同根）：
  finetune_tw/data/tw_stocks.db   — 股價 DB（每次 sandbox 重新下載）
  finetune_tw/outputs/            — tokenizer & predictor checkpoint
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _imports():
    import marimo as mo
    import os, sys, subprocess, tarfile, dataclasses, datetime
    from pathlib import Path
    return mo, os, sys, subprocess, tarfile, dataclasses, datetime, Path


@app.cell
def _setup(mo, os, sys, subprocess, tarfile, Path):
    REPO_URL = "https://github.com/j835111/Kronos.git"
    REPO_DIR = str(Path.home() / "Kronos")
    FTW_TAR  = str(Path.home() / "finetune_tw.tar.gz")

    if not Path(REPO_DIR).exists():
        subprocess.run(["git", "clone", "--depth=1", REPO_URL, REPO_DIR], check=True)

    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    if Path(FTW_TAR).exists():
        with tarfile.open(FTW_TAR, "r:gz") as tar:
            tar.extractall(REPO_DIR, filter="data")
    elif not Path("finetune_tw").exists():
        raise FileNotFoundError(
            "Upload finetune_tw.tar.gz via the sidebar file browser first."
        )

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "yfinance", "pyyaml", "tqdm", "safetensors", "huggingface_hub"],
        check=True,
    )

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "⚠️ CPU only"
    mo.md(f"### ✅ 環境就緒  \n**GPU:** `{gpu}`")
    return REPO_DIR, FTW_TAR, torch


@app.cell
def _download(mo, Path, REPO_DIR, datetime):
    DB_DST = str(Path(REPO_DIR) / "finetune_tw" / "data" / "tw_stocks.db")
    Path(DB_DST).parent.mkdir(parents=True, exist_ok=True)

    if not Path(DB_DST).exists():
        from finetune_tw.download_data import download as _dl
        from finetune_tw.fetchers.yfinance_fetcher import get_twse_symbol_list as _syms
        _symbols = ["^TWII"] + _syms()
        _dl(db_path=DB_DST, symbols=_symbols,
            start="2015-01-01", end=str(datetime.date.today()), source="yfinance")

    import sqlite3
    with sqlite3.connect(DB_DST) as _c:
        _rows = _c.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
        _n    = _c.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]

    mo.md(f"### ✅ 資料就緒：{_n} 支股票，{_rows:,} 筆")
    return DB_DST,


@app.cell
def _config(mo, dataclasses, DB_DST):
    from finetune_tw.config import Config

    cfg = Config.from_yaml("finetune_tw/configs/config_tw_daily.yaml")
    cfg = dataclasses.replace(cfg,
        db_path=DB_DST,
        output_dir=str(Path(REPO_DIR) / "finetune_tw" / "outputs"),
        tokenizer_epochs=30,
        basemodel_epochs=20,
        batch_size=128,        # RTX Pro 6000 96GB — 可以比 T4 大很多
        num_workers=4,
        save_steps=500,
        log_interval=100,
    )

    mo.md(f"""
    ### ⚙️ 訓練設定
    | 參數 | 值 |
    |------|-----|
    | tokenizer_epochs | `{cfg.tokenizer_epochs}` |
    | basemodel_epochs | `{cfg.basemodel_epochs}` |
    | batch_size | `{cfg.batch_size}` |
    | train / val split | `{cfg.train_end_date}` / `{cfg.val_end_date}` |
    """)
    return cfg,


@app.cell
def _train_tokenizer(mo, cfg):
    print("=" * 60)
    print("STAGE 1 / 3 — Tokenizer Training  [fp32, no autocast]")
    print("=" * 60)
    from finetune_tw.train_tokenizer import run_training as _train_tok
    _train_tok(cfg)
    mo.md("### ✅ Stage 1 完成：Tokenizer")
    return


@app.cell
def _train_predictor(mo, cfg):
    print("=" * 60)
    print("STAGE 2 / 3 — Predictor Training  [fp32, no autocast]")
    print("=" * 60)
    from finetune_tw.train_predictor import run_training as _train_pred
    _train_pred(cfg)
    mo.md("### ✅ Stage 2 完成：Predictor")
    return


@app.cell
def _backtest(mo, cfg, subprocess, sys, Path):
    import yaml

    bt_cfg_path = "/tmp/config_molab_backtest.yaml"
    with open(bt_cfg_path, "w") as _f:
        yaml.dump({
            "db_path": cfg.db_path,
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
        }, _f)

    subprocess.run(
        [sys.executable, "finetune_tw/backtest.py", "--config", bt_cfg_path],
        check=False,
    )

    img_path = Path(cfg.output_dir) / cfg.exp_name / "backtest_result.png"
    mo.image(str(img_path)) if img_path.exists() else mo.md("⚠️ backtest 圖表未產生")


if __name__ == "__main__":
    app.run()
