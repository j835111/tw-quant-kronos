# Autoresearch Improve — 執行摘要

**目標：** 改進 Kronos 台股預測模型（Round 0 fine-tune）  
**ICP：** 台股量化動能策略研究者，open-to-open 執行  
**深度：** Standard (15 iterations)  
**狀態：** BOUNDED (15/15)

## 研究統計

| 指標 | 值 |
|------|-----|
| 總迭代次數 | 15 |
| 新 insight | 11 |
| 延伸 insight | 0 |
| 類別覆蓋 | 5/5 |
| HIGH 信心 | 6 |
| MEDIUM 信心 | 4 |
| LOW 信心 | 1 |

## 最重要發現

1. **Label Horizon Paradox (ICML 2026, arxiv:2602.03395)** 精確描述了我們的問題：訓練標籤（token CE，隱含 close-to-close）≠ 推理目標（open-to-open）。論文提供理論基礎和 bi-level 解決方案。

2. **Ranking losses > pointwise losses (arxiv:2510.14156, CIKM 2025)**：我們的 token CE 本質是 pointwise，而選股任務需要 pairwise/listwise 損失。

3. **預測 high-low 範圍可以直接解決 MaxDD 問題**：模型已在輸出這個資訊但完全沒有用。ATR position sizing 是成熟的業界實踐，組合到 Kronos 的前向 ATR 是非常有機的改善。

## 建議改進（按優先級）

| 優先 | 名稱 | 信心 | 重訓？ | 主要效果 |
|------|------|------|--------|---------|
| #1 | M1: ATR position sizing | HIGH | ❌ | MaxDD 35%→20% |
| #2 | M2: Open-to-open IC | HIGH | ✅ | Sharpe 1.356→1.5+ |
| #3 | M3: Volume filter | MEDIUM | ❌ | 選股品質改善 |
| #4 | N1: Ranking loss | HIGH | ✅ | IC 0.04→0.06+ |
| #5 | N2: 擴大 IC 驗證集 | MEDIUM | ✅ | Early stop 穩定性 |

## 輸出檔案

- `research-findings.md` — 11 個 insight，附引用來源與信心等級
- `improvement-plan.md` — 8 項改進方案，3 tier 分級，含實作細節
- `handoff.json` — 機器可讀的 findings 摘要
