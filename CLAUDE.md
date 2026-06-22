# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kronos is a decoder-only foundation model for financial K-line (candlestick) sequences, pre-trained on data from 45+ global exchanges. It uses a two-stage architecture: a specialized tokenizer quantizes OHLCV data into discrete tokens, then an autoregressive Transformer performs forecasting.

Published at AAAI 2026. Models are hosted on HuggingFace under `NeoQuasar/`.

## Commands

### Install
```bash
pip install -r requirements.txt
# For qlib-based finetuning also:
pip install pyqlib
```

### Run tests
```bash
pytest tests/
# Single test:
pytest tests/test_kronos_regression.py::test_kronos_predictor_regression
# Tests download models from HuggingFace on first run — requires internet access
```

### Finetune (CSV data)
```bash
# Edit finetune_csv/configs/*.yaml first, then:
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_ali09988_candle-5min.yaml
```

### Finetune (Qlib / A-share)
```bash
# Edit finetune/config.py paths, then:
python finetune/qlib_data_preprocess.py
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
python finetune/qlib_test.py --device cuda:0
```

### Finetune (Taiwan stocks / finetune_tw)

選擇對應你硬體的 config：

| 環境 | Config | AMP |
|------|--------|-----|
| MoLab RTX Pro 6000 (96 GB) | `config_tw_daily_rtx6000.yaml` | bf16 |
| Colab T4 (16 GB) | `config_tw_daily_t4.yaml` | fp16 + GradScaler |
| 其他 / 安全預設 | `config_tw_daily.yaml` | 關閉 |

```bash
# MoLab 訓練（RTX Pro 6000）
CONFIG=finetune_tw/configs/config_tw_daily_rtx6000.yaml

# Colab T4
CONFIG=finetune_tw/configs/config_tw_daily_t4.yaml

# 1. Download TWSE daily data into SQLite:
python -m finetune_tw.download_data --config $CONFIG
# Incremental update:
python -m finetune_tw.download_data --config $CONFIG --update

# 2. Fine-tune tokenizer:
python -m finetune_tw.train_tokenizer --config $CONFIG

# 3. Fine-tune predictor (frozen tokenizer, bf16 AMP on RTX, supports resume):
python -m finetune_tw.train_predictor --config $CONFIG

# 4. Backtest (top-K hold strategy):
python -m finetune_tw.backtest --config $CONFIG

# 5. [Optional] Grid-search top_k × hold_days (inference once, CPU sweep):
python -m finetune_tw.grid_search_backtest --config $CONFIG --model round0 \
    --top_k_list 10 20 30 50 --hold_days_list 3 5 7 10

# For Colab: open finetune_tw/colab_setup.ipynb
```

### Finetune (Taiwan stocks / MoLab)
使用 `finetune_tw/molab_train.py`，在 [molab.marimo.io](https://molab.marimo.io) 開啟：

1. 新建 notebook，點右上角 notebook specs → 啟用 GPU
2. 執行所有 cell（Run All）

**持久化儲存：`/marimo/Kronos/`**（repo 根目錄，sandbox 重啟後資料保留）
- `/marimo/Kronos/finetune_tw/data/tw_stocks.db` — 股價 DB，第一次下載後永久保留
- `/marimo/Kronos/finetune_tw/outputs/` — tokenizer & predictor checkpoint，重啟後自動 resume

> **注意**：VM filesystem（`/home/marimo/`、`/tmp/` 等）每次 sandbox 重啟會清空。`/mnt/first/` 換 PV 時也會清空，不可靠。資料一律放在 `/marimo/Kronos/` 下。

### Finetune (Taiwan stocks / Colab)
使用 `finetune_tw/colab_setup.ipynb`，按 cell 順序執行：

1. **掛載 Google Drive** — 資料與 checkpoint 持久化至 `MyDrive/Kronos_TW/`
2. **Clone repo** — `https://github.com/shiyu-coder/Kronos`，並將 `finetune_tw/data` 與 `finetune_tw/outputs` symlink 至 Drive
3. **安裝依賴** — `requirements.txt` + `yfinance pyyaml tqdm`
4. **下載台股資料** — `python -m finetune_tw.download_data --source auto --start 2015-01-01`
5. **訓練 tokenizer** — `python -m finetune_tw.train_tokenizer --config finetune_tw/configs/config_tw_daily_t4.yaml`
6. **訓練 predictor** — `python -m finetune_tw.train_predictor --config finetune_tw/configs/config_tw_daily_t4.yaml`（支援 resume，重啟 session 後重跑即可）
7. **回測** — `python -m finetune_tw.backtest --config finetune_tw/configs/config_tw_daily_t4.yaml`

> **注意**：Colab T4 使用 `config_tw_daily_t4.yaml`（`amp_dtype: "fp16"`，Turing 原生 fp16 TensorCore + GradScaler）；MoLab RTX Pro 6000 改用 `config_tw_daily_rtx6000.yaml`（bf16 啟用）。Kronos-base pretrained 權重從 `NeoQuasar/Kronos-base` 自動下載。

## Iterative Improvement Loop (autoresearch + Colab)

針對 `finetune_tw` 台股模型的持續優化流程，循環至回測指標收斂為止。

### 流程總覽

```
[分析現狀] → [提出調整] → [Colab 訓練] → [評估結果] → 收斂? → 結束
                 ↑                                          ↓ 否
                 └──────────────────────────────────────────┘
```

### Step 1：分析與提出調整（autoresearch）

```
/autoresearch:improve
```

Claude 會：
- 讀取 `finetune_tw/configs/config_tw_daily.yaml` 與回測輸出
- 對照回測指標（Annual Return、Sharpe、Max Drawdown）找出瓶頸
- 提出具體調整方案：超參數、資料前處理、模型架構選項

調整範圍優先順序（由低風險到高風險）：
1. **Config 超參數**：`lookback_window`、`top_k`、`hold_days`、`pred_len`、`lr`、`epochs`
2. **資料**：訓練/驗證切割日期、clip 值、symbol 篩選條件
3. **模型架構**：切換 `pretrained_predictor`（Kronos-small / Kronos-base）

### Step 2：在 Colab 執行訓練

開啟 `finetune_tw/colab_setup.ipynb`，從 Drive 取得最新 config 後依序執行：

```
Cell 1: 掛載 Drive
Cell 2: Clone / pull repo（確保 config 更新已同步）
Cell 3: 安裝依賴
Cell 4: 下載/更新台股資料（--update 模式）
Cell 5: train_tokenizer.py
Cell 6: train_predictor.py（自動 resume）
Cell 7: backtest.py → 輸出 metrics 與圖表至 Drive
```

### Step 3：評估結果（autoresearch:evals）

```
/autoresearch:evals
```

Claude 會：
- 讀取 `finetune_tw/outputs/` 內的回測 metrics
- 比較本輪與前輪的 Sharpe、Annual Return、Max Drawdown
- 判斷是否收斂（連續 2 輪改善 < 1% 視為收斂）

### Step 3.5：策略參數優化（grid_search_backtest）

在調整模型之前，**先跑 grid search** 確認問題是否出在策略參數而非模型能力：

```bash
python -m finetune_tw.grid_search_backtest --config $CONFIG \
    --model round0 --top_k_list 10 20 30 50 --hold_days_list 3 5 7 10
```

> **重要發現（2026-06-22）**：Round 0 模型用 `top_k=10, hold_days=3` 可達 Sharpe 1.84、年化 80%，
> 而原始設定 `top_k=20, hold_days=5` 只有 Sharpe 1.19、年化 40%。
> **策略參數的影響遠大於重新訓練。** Round 1 和 Round 2 的 retraining 是在解錯問題。

### 收斂條件

| 指標 | 目標 | 說明 |
|------|------|------|
| Sharpe Ratio | ≥ 1.5 | 相對 ^TWII 基準 |
| Annual Return | > 15% | 測試集 2024-07-01 起 |
| Max Drawdown | < 20% | |
| 連續改善幅度 | < 1% | 連續 2 輪則停止 |

### 當前最佳設定（config_tw_daily_rtx6000.yaml）

| 參數 | 值 | 來源 |
|------|-----|------|
| `top_k` | 10 | grid search 最佳 |
| `hold_days` | 3 | grid search 最佳 |
| `pretrained_predictor` | `j835111/kronos-tw-finetune` | Round 0 fine-tuned |
| `hf_revision` | `round-0` | HF branch |

### 注意事項

- 每輪調整前先用 `git commit` 保存當前 config，方便回滾
- Colab 斷線重連後直接重跑 Cell 6（predictor 支援 resume）
- 調整 tokenizer 超參數需重跑 Cell 5+6；只調 predictor/backtest 只需重跑 Cell 6+7
- **先跑 grid_search_backtest 再決定是否要 retrain**

## Architecture

### Core model (`model/`)
- `model/kronos.py` — all primary classes, exported via `model/__init__.py`:
  - `KronosTokenizer`: Encoder-decoder Transformer with Binary Spherical Quantization (BSQuantizer). Converts raw OHLCV windows into two-level discrete token indices (`s1_ids`, `s2_ids`).
  - `Kronos`: Autoregressive decoder-only Transformer that operates on the token space. Predicts next tokens given a sequence of `(s1_ids, s2_ids)` pairs plus optional time-stamp embeddings.
  - `KronosPredictor`: High-level inference wrapper. Handles normalization, context windowing, autoregressive decoding, and inverse normalization. Entry point for all prediction tasks.
- `model/module.py` — lower-level building blocks: `TransformerBlock`, `BSQuantizer`, attention layers.

### Two-stage inference flow
```
raw OHLCV DataFrame
  → normalize (per-window z-score)
  → KronosTokenizer.encode() → (s1_ids, s2_ids)
  → Kronos.decode_s1() / decode_s2() autoregressive loop
  → KronosTokenizer.decode() → normalized OHLCV
  → inverse normalize → forecast DataFrame
```

### Fine-tuning pipelines
- `finetune/` — Qlib-based pipeline for Chinese A-share daily data. Config in `finetune/config.py`. Trains tokenizer first, then predictor.
- `finetune_csv/` — CSV-based pipeline for any OHLCV data. YAML config in `finetune_csv/configs/`. `train_sequential.py` orchestrates both stages.
- `finetune_tw/` — Taiwan stock fine-tuning pipeline. Fetches TWSE daily OHLCV data into a local SQLite DB, fine-tunes tokenizer then predictor on multi-stock data, and runs a top-K backtester. Supports Colab via `colab_setup.ipynb`. Config in `finetune_tw/configs/config_tw_daily.yaml`.

## Model Zoo

| Model | Tokenizer | Context | Params |
|-------|-----------|---------|--------|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M |

`Kronos-small` + `Kronos-Tokenizer-base` is the standard pairing for most examples and tests.

## Data Format

All CSV inputs require columns: `timestamps`, `open`, `high`, `low`, `close`. `volume` and `amount` are optional (filled with 0 if absent). Reference data in `finetune_csv/data/` for format examples.

## Key Notes

- `max_context=512` is the hard limit for `Kronos-small`/`Kronos-base`; `KronosPredictor` auto-truncates longer contexts.
- Regression tests pin specific HuggingFace commit revisions (`MODEL_REVISION`, `TOKENIZER_REVISION`) to ensure determinism.
- `finetune/config.py` contains hardcoded paths marked with `TODO` that must be updated before running.
- Code comments in `finetune/` were AI-generated and may contain inaccuracies; treat the code as authoritative.
