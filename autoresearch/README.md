# autoresearch/ 索引

`/autoresearch:improve` 各次產出的改進計劃(improvement-plan、PRD、research-findings、summary)。
每個 `improve-*` 目錄對應一次分析,其提案在後續 Round 中驗證;完整結論與失敗分析見
[docs/kronos-tw-round-history.md](../docs/kronos-tw-round-history.md)(含逐方法對照表)。

## 計劃 → 驗證輪次 → 結果

| 計劃目錄 | 主要提案 | 驗證輪次 | 結果 |
|---|---|---|---|
| `improve-260622-0042/` | IC-IR@h5 early stopping、驗證集 300×20、從 Round 0 起點、Warmup+Cosine | Round 2 | ❌ Sharpe 1.14,輸 Round 0;僅「從 Round 0 起點」確立為必要條件 |
| `improve-260626-1240/` | ATR position sizing、volume filter(Week 1);open-to-open IC early stop、驗證集 500×40(Round 3) | Week 1 / Round 3 | ❌ 策略層改動全數 no-op;Round 3 大幅退步(Sharpe 0.50) |
| `improve-260629-1426/` | FPT Selective Freeze、IC-IR@h1 early stopping、Extended Warmup | Round 4 | ❌ best epoch=1,Sharpe 1.24 仍輸 Round 0,確認 Round 0 為 fine-tuning 局部最優 |
| `improve-260701-1512/` | 凍結 Kronos 當特徵抽取器 + XGBoost LambdaRankIC(M1) | Round 6 | ⚠️ 初版 Sharpe 0.34(regime 依賴);經 Batch 1–3c 修正 + Direction 2 融合後達 **Sharpe 1.5434,成為 production** |

Round 5(Pretrained 重啟 + Auxiliary Ranking Loss)取材自 260622/260626/260629 三份計劃的 ranking loss 提案(N1),結果 Sharpe 0.98,未轉化為回測改善。

## 評估資料

- `tw-evals/finetune-tw-results.tsv` — 各輪回測指標彙整表(Sharpe / Ann / MaxDD),與 round-history 的彙整章節對應。

## 慣例

- 目錄名格式:`improve-<YYMMDD>-<HHMM>`,YY 為西元末兩碼(260622 = 2026-06-22)。
- 結束一輪 research 分支前,先確認 `tw-evals/finetune-tw-results.tsv` 與 round-history 已在 master 更新(見 CLAUDE.md branch strategy)。
