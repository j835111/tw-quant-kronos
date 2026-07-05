# tw-quant-kronos

台股量化研究專案 — 以 [Kronos](https://github.com/shiyu-coder/Kronos)(AAAI 2026,金融 K 線基礎模型)為底層,對台股日線資料 fine-tune,並建立回測與每日選股信號的完整 pipeline。

> 本 repo fork 自 `shiyu-coder/Kronos`,但主軸已轉為台股量化研究;上游模型程式碼(`model/`)以 vendored library 形式保留並持續沿用。上游原始說明見 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。

## 現行 Production 基準

| 項目 | 值 |
|------|-----|
| 模型 | Kronos-base fine-tune(`j835111/kronos-tw-finetune`)+ XGBoost full/raw 雙模型 |
| 融合方式 | 靜態 Z-Score ensemble,`w = 0.6` |
| 策略參數 | `top_k = 10` |
| 回測 Sharpe | **1.5434**(Round 6 Direction 2,取代先前所有版本) |
| 每日信號 | `scripts/run_signal_today_ensemble.sh` |

```bash
# 每日信號產生(更新 DB → 產出選股清單)
bash scripts/run_signal_today_ensemble.sh

# 舊版單模型信號(Round 0 baseline)
bash scripts/run_signal_today.sh
```

## 專案結構

| 目錄 | 內容 | 來源 |
|------|------|------|
| `finetune_tw/` | 台股 pipeline:資料下載(TWSE → SQLite)、tokenizer/predictor fine-tune、回測、grid search、每日信號 | 本專案 |
| `scripts/` | 每日信號腳本、訓練輔助腳本 | 本專案 |
| `autoresearch/` | 迭代優化實驗紀錄與評估結果(`tw-evals/finetune-tw-results.tsv`) | 本專案 |
| `docs/` | 研究歷程(`kronos-tw-round-history.md`)、實驗計畫、上游 README | 本專案 |
| `tests/` | 回歸測試與 `finetune_tw` 單元測試 | 本專案擴充 |
| `model/` | Kronos 核心模型(tokenizer + autoregressive Transformer),含本地修補 | 上游(vendored) |
| `finetune/`, `finetune_csv/`, `examples/`, `webui/`, `figures/` | 上游原始 pipeline 與範例,未修改 | 上游 |

## 快速開始(台股 pipeline)

```bash
pip install -r requirements.txt

CONFIG=finetune_tw/configs/config_tw_daily.yaml

# 1. 下載/更新台股日線資料(SQLite)
python -m finetune_tw.download_data --config $CONFIG            # 首次
python -m finetune_tw.download_data --config $CONFIG --update   # 增量

# 2. Fine-tune(tokenizer → predictor)
python -m finetune_tw.train_tokenizer --config $CONFIG
python -m finetune_tw.train_predictor --config $CONFIG

# 3. 回測(top-K hold 策略)
python -m finetune_tw.backtest --config $CONFIG

# 4. 策略參數 grid search(推論一次,CPU sweep)
python -m finetune_tw.grid_search_backtest --config $CONFIG --model round0 \
    --top_k_list 10 20 30 50 --hold_days_list 3 5 7 10
```

訓練環境選擇(config 對應硬體):

| 環境 | Config |
|------|--------|
| MoLab RTX Pro 6000(96 GB) | `config_tw_daily_rtx6000.yaml`(bf16) |
| Colab T4(16 GB) | `config_tw_daily_t4.yaml`(fp16 + GradScaler) |
| 其他 / 安全預設 | `config_tw_daily.yaml` |

Colab 流程見 `finetune_tw/colab_setup.ipynb`;MoLab 流程見 `finetune_tw/molab_train.py` 與 `CLAUDE.md`。

## 研究歷程

完整輪次歷史與結論記錄在:

- [`docs/kronos-tw-round-history.md`](docs/kronos-tw-round-history.md) — 各輪實驗敘述與決策
- [`autoresearch/tw-evals/finetune-tw-results.tsv`](autoresearch/tw-evals/finetune-tw-results.tsv) — 回測指標總表

重點結論:

- **策略參數的影響遠大於 retraining**(Round 0 + grid search 即達 Sharpe 1.84 in-sample)。
- Round 6 Direction 2:Kronos 信號與 XGBoost(full/raw 特徵)做靜態 Z-Score 融合,`w=0.6` 為現行 production(Sharpe 1.5434)。
- Stacking、MC dropout、backbone 替換(Direction 1)、rank horizon 解耦等方向均已驗證無效或有害,詳見 round history。

## 底層模型:Kronos

Kronos 是 decoder-only 的金融 K 線基礎模型,以 45+ 交易所資料預訓練,採兩階段架構:tokenizer 將 OHLCV 量化為離散 token,再由自回歸 Transformer 預測。發表於 AAAI 2026([arXiv:2508.02739](https://arxiv.org/abs/2508.02739))。

| Model | Tokenizer | Context | Params |
|-------|-----------|---------|--------|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M |

預訓練權重託管於 HuggingFace [`NeoQuasar/`](https://huggingface.co/NeoQuasar);本專案 fine-tuned 權重在 [`j835111/kronos-tw-finetune`](https://huggingface.co/j835111/kronos-tw-finetune)。

## License

沿用上游 [MIT License](LICENSE)。

## ⚠️ 免責聲明

本專案僅供研究用途,所有回測結果與選股信號不構成投資建議。
