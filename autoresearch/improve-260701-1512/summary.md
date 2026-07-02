# Summary — Kronos TW Round 6 研究
> 日期：2026-07-01 | 迭代：10 | 狀態：SATURATED（核心新方向已識別）

## 研究統計
- 總迭代：10
- 全新 insight：8
- 覆蓋類別：5/5（ICP challenges, Competitor gaps, Market trends, UX & experience, Revenue & growth）
- 飽和狀態：YES（後 3 輪 net-new insights < 2）

## 核心結論

五輪 fine-tuning 失敗的**真正原因**（文獻確認）：
- 研究（arXiv:2511.18578）用 20 億筆全球股市資料確認：pretrained TSFM 在金融回報預測的 fine-tuning 環境下系統性失敗
- 這不是超參數問題，是架構性問題

## 三個全新方向

| 優先 | 方向 | 核心變化 | 工程量 | 信心 |
|------|------|---------|--------|------|
| 🔴 必做 | **M1: Kronos Embedding + LambdaRankIC (XGBoost)** | 完全不 fine-tune，用 hidden states 訓練 XGBoost | 1-2 天 | HIGH |
| 🟠 可選 | **N1/N2: L2-SP / MoFO 正則化** | 防止 fine-tuning 退化（若想繼續 fine-tune 路線） | 2-6 小時 | MEDIUM |

## 輸出檔案
- `research-findings.md` — 8 個 insights，每個含信心等級與來源
- `improvement-plan.md` — M1/M2/N1-N3 完整 PRD

## 下一步行動

**立即可做**（不需 GPU）：
1. 實作 `extract_embeddings.py`：凍結 Kronos-base，批次提取所有 1091 支股票的 hidden states
2. 實作 `train_xgb_lambdarank.py`：訓練 XGBoost，以 LambdaRankIC loss 直接優化 Rank IC

**RunPod（$5 預算）**：
1. 跑 M2（pretrained restart + close IC-IR@h1，無 ranking loss）對照
