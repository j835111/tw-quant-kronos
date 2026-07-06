# docs/ 文件導覽

tw-quant-kronos 的文件分四類:歷史主軸、單輪深度報告、未來方向、操作指南。
研究過程的唯一敘事主軸是 `kronos-tw-round-history.md`,其他文件皆為它引用的附件。

## 歷史主軸

- [kronos-tw-round-history.md](kronos-tw-round-history.md) — Round 0–6 完整微調歷史:每輪的起點、調整、訓練歷程、回測結果與失敗分析,附「Autoresearch 方法完整對照表」(已驗證 vs 未驗證)。**看這份就能掌握全部研究脈絡。**

## 單輪深度報告(`research/`)

被 round-history 摘要引用的完整分析,依對應輪次命名:

- [research/round0-1-predictor-retrain-analysis.md](research/round0-1-predictor-retrain-analysis.md) — Round 0/1 對比 pretrained 的預測品質體檢(eval_forecast、IC 衰減),解釋 Round 1 失敗的三個根本原因。
- [research/round6-embedding-xgb-lambdarank-improvements.md](research/round6-embedding-xgb-lambdarank-improvements.md) — Round 6 全過程:源碼缺陷診斷、Batch 1–3c 修正實驗、Direction 2 融合(最終 production,Sharpe 1.5434)的完整記錄。
- [research/round6-artifact-evaluation.md](research/round6-artifact-evaluation.md) — Round 6 產物(embeddings、xgb model)的獨立實證評估。

## 未來方向(`research-directions/`)

- [research-directions/kronos-external-research-directions.md](research-directions/kronos-external-research-directions.md) — 外部調研(GitHub/HF/arXiv)整理出的後續可執行方向(2026-07-01)。
- [research-directions/related-work-survey.md](research-directions/related-work-survey.md) — 金融時序基礎模型相關專案與論文調研(2026-07-01)。

尚未驗證的具體方法清單見 round-history 末段的「未驗證」對照表。

## 操作指南

- [TRADING_GUIDE.md](TRADING_GUIDE.md) — 台股實盤操作指南。⚠️ 內容尚停留在 Batch 3c `full` 單模型,待更新至 Direction 2 Z-Score 融合(w=0.6)production 現況;production 腳本以 `scripts/run_signal_today_ensemble.sh` 為準。

## 其他

- [UPSTREAM_README.md](UPSTREAM_README.md) — 上游 Kronos 專案原始 README(vendored)。
- `superpowers/` — 各次開發任務的執行計劃(`plans/`)與設計文件(`specs/`),檔名含日期,依時間排序。
- `assets/` — 文件用圖片。

## 相關目錄

- `../autoresearch/` — `/autoresearch:improve` 產出的改進計劃與 PRD,含各輪回測指標彙整表(`tw-evals/finetune-tw-results.tsv`),索引見 [../autoresearch/README.md](../autoresearch/README.md)。
