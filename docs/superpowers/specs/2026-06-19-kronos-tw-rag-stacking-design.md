# Kronos-TW：微調 + 類比檢索 + Stacking 集成架構設計

> **狀態**：藍圖（next-phase design）。撰寫於 2026-06-19，資料尚在下載，未有 baseline 回測數字。
> **適用模組**：`finetune_tw/`
> **一句話**：把原始「微調 + token-prefix RAG + 0.6/0.4 加權」重構為「微調主幹 + 類比檢索特徵 + walk-forward stacking meta-model」，每一步都用 **Rank-IC** 量增量貢獻。

---

## 1. 背景

Kronos 本質是「OHLCV tokenizer + 自回歸 Transformer」：tokenizer 把 K 線量化成離散 token，predictor 在 token 空間自回歸預測。`Kronos-small` / `Kronos-base` 的 context 硬上限是 **512**（`KronosPredictor` 會自動截斷更長的脈絡）。官方提供 **tokenizer → predictor 兩階段微調**流程，並明確提醒 repo 內的 demo backtest **不是 production trading system**，正式落地需自行處理成本、滑價、風險與組合建構。

**參考來源**
- GitHub README — https://github.com/shiyu-coder/Kronos#readme
- arXiv paper — https://arxiv.org/abs/2508.02739
- 程式：`model/kronos.py`（`KronosPredictor` 推理流程）、`finetune_tw/backtest.py`、`finetune_tw/dataset.py`

**現況（`finetune_tw`）**：TWSE 日線多檔個股存於 SQLite；`backtest.py` 對每檔股票預測未來 `pred_len=10` 日報酬（`pred_close[-1]/now_close-1`），按日做**橫斷面排序**取 `top_k=20`，持有 `hold_days=5` 再平衡，對標 `^TWII`。

---

## 2. 範圍與非目標（Scope & Non-goals）

**本藍圖鎖定**：日線 TWSE 多股、橫斷面 **top-K 選股排序**（沿用現況 `finetune_tw`）。

- 同一套方法論可移植到**單一商品的日線方向擇時**（如對 `^TWII` 或台指期日線做多空判斷，走 `finetune_csv` 管線）作為未來 pivot；本藍圖不實作，但 Layer 0 的任務定義會保留 horizon 抽象（N 日），方便日後切換。
- **本階段非目標**：production 即時交易系統、即時下單。交易摩擦層（Layer 4）只定義介面，不在前期投入。

---

## 3. 核心校準：精準 = Rank-IC，不是 MSE

決策是「每天把所有股票按預測未來報酬排序，取 top-20」。因此真正要優化的是**同一天跨股票的相對順序**，而非單點位的絕對精準度。

**主指標**
- **Rank-IC**（預測報酬 vs 實現報酬的橫斷面 Spearman 相關）— 開發期的早期訊號
- **Top/Bottom decile 報酬差**
- **Sharpe / Max Drawdown vs `^TWII`** — 最終驗收

這條校準會修正原架構「LightGBM 補足絕對點位精準度」的傾向：對選股排序，point-wise OHLCV 的 MSE 幾乎不重要。

---

## 4. 整體架構

```
raw OHLCV (SQLite, point-in-time)
   │
   ▼
[Layer 0] 任務定義 + walk-forward / purged split + baseline ladder
   │
   ▼
[Layer 1] Kronos 主幹
   zero-shot → predictor-only FT → (tokenizer+predictor) FT
   訊號 = MC 平均的未來報酬 + 樣本分位數 + dispersion
   │
   ▼
[Layer 2] Analog Engine（RAG-as-features，不進 Kronos context）
   k-NN 檢索 point-in-time 歷史窗 → 類比歷史統計特徵
   │
   ▼
[Layer 3] LightGBM meta-model（stacking，cross-sectional，rank 目標）
   features = Kronos 訊號 + analog 統計 + 技術/量價/市場相對/regime
   target   = 未來 N 日報酬 / 方向機率 / risk-adjusted signal
   │
   ▼
   按日排序取 top-K
   │
   ▼
[Layer 4] 交易系統（最後才做）：成本 / 滑價 / 換倉限制 / 風控
   │
   ▼
   PnL / Sharpe / MDD
```

**設計原則**：三層各有單一職責、介面清楚、可獨立測試與替換。Kronos 訊號、analog 統計、meta-model 之間只透過「特徵表」溝通，任何一層可單獨換掉而不影響其他層。

---

## 5. 逐層設計

### Layer 0 — 任務定義與資料切割（先做，最關鍵）

- **任務**：預測未來 `pred_len`（日線預設 10 日）報酬；輸出按日橫斷面排序取 `top_k`。Horizon 抽象保留多目標可能（如同時看 N=5/10/20 日）。
- **Walk-forward + purged / embargo split**：train / val / test 之間必須留時間 gap，且 gap **≥ `lookback_window + predict_window`**（現況 90 + 10），否則滑動視窗重疊會把未來洩漏進訓練集。這對日線同樣成立（lookback=90 的窗彼此高度重疊）。
- **Baseline ladder（每一步都要比）**：
  1. zero-shot Kronos
  2. predictor-only fine-tune（凍結 tokenizer）
  3. tokenizer + predictor 全微調
  4. LightGBM-only（純技術/量價特徵，無 Kronos）
  5. naive baseline（動能 / 反轉）
  6. kNN-analog-only（只用 Layer 2 的類比統計）
- 沒有這個階梯，無法判斷增益究竟來自 Kronos、來自特徵工程、還是來自 analog。

### Layer 1 — Kronos 主幹（微調）

- **順序：先別動 tokenizer。** 依序跑 zero-shot → predictor-only → tokenizer+predictor，逐步驗證每一步是否真的帶來 Rank-IC 增益。
  - 理由：tokenizer 一改，token 空間就變，資料不夠大時容易過擬合；單一市場數年資料看似多，但 **regime 數量未必足夠**。
  - 算力：Colab T4 上 `Kronos-base` 全量微調慢且吃 VRAM；優先 `Kronos-small` 或 predictor-only。
- **訊號產生（修正現況）**：`backtest.py` 目前用 `T=1.0, top_k=1, sample_count=1`，是近乎貪婪的**單樣本**，雜訊大。改為 `sample_count ≥ 20~30` 取**期望報酬**，並輸出**分位數 + sample dispersion** 給 Layer 3 當特徵。
- **Checkpoint 選擇**：用 **val Rank-IC** 選 `best_model`，不要用重建 / CE loss（重建最佳 ≠ 排序最佳）。
- **防洩漏**：在各 walk-forward fold 內訓練，fold 之間遵守 embargo。

### Layer 2 — Analog Engine（RAG 改成檢索特徵，**不進 Kronos context**）

**為什麼不直接 prepend 歷史 token 到 Kronos：**
- Kronos 沒有明確的「片段分隔 token」，會把 context 當成**連續時間序列** → 接縫處產生假連續關係，等於餵給模型虛構的近期歷史（out-of-distribution）。
- 各歷史片段的 **normalization、timestamp、session 斷點**都不一致，縫起來更假。
- small/base 只有 **512 context**；3 段 60 根就吃掉 180 根，會**擠掉真正最近的市場狀態**——而那正是排序訊號最依賴的部分。

**正確做法 — analog feature engine：**
- **檢索鍵**：log-return、range（高低波幅）、volume z-score、日曆季節性（weekday / 月份，對應 `dataset.py` 的 stamps）、或 Kronos 的 token / encoder embedding（z-score 後比形狀，不比絕對價位）。
- **檢索庫（point-in-time，鐵律）**：每個 walk-forward fold 的檢索庫**只能包含當時以前**的資料；命中窗的未來結果必須在預測時點 T 已實現（窗結束 ≤ T − `pred_len`）。
- **輸出特徵**（交給 Layer 3，不送進 Kronos）：
  - 相似片段後續 N 日的 forward return **分位數**
  - **上漲機率**
  - **最大順向 / 逆向波動**
  - **量能擴張後的續航率**
  - **regime 標籤** + **analog dispersion**（信心）
- **好處**：保留你要的 RAG 價值——「看歷史上類似劇本怎麼演」、對極端行情（暴跌/暴漲）特別有用——但完全不破壞 Kronos 的自回歸假設。
- **🚩 最大紅線**：洩漏 = 假性 Sharpe 爆表。檢索庫的 point-in-time 切割是不可妥協的。

### Layer 3 — LightGBM meta-model（stacking）

**不寫死 `0.6 * Kronos + 0.4 * LGBM`。** 讓 meta-model 學「**何時該信 Kronos、何時該打折**」。

- **Features**：
  - Kronos：預測均值、分位數、sample dispersion、方向
  - Analog：Layer 2 的全部類比統計
  - 技術 / 量價：MA gap、RSI、布林 %b、量能爆發、動能、波動率
  - **市場相對（補 Kronos 缺的橫斷面脈絡，CP 值最高）**：個股報酬 − `^TWII`、產業內排名
  - Regime：波動率水準、隔日跳空、除權息 / 財報公布 proximity、大盤多空 regime、重大事件 proximity
- **Target（非單純點位殘差）**：未來 N 日報酬 / 方向機率 / post-cost expected return / 分位數風險。
- **排序對齊**：用 `lambdarank`（或回歸後看 Rank-IC），並**按日分組（cross-sectional）**訓練——對齊「每天排序選股」的真實決策。
- **防過擬合（關鍵）**：Kronos 與 analog 特徵必須以 **walk-forward / out-of-fold** 方式產生，否則 stacker 會過擬合 Kronos 的 in-sample 怪癖。這也是本層**唯一**的主要計算成本（Kronos 需在訓練期跑一遍推理）。

### Layer 4 — 交易系統（最後才做）

加入成本、滑價、手續費、換倉限制、最大持倉、停損 / 停利、波動率縮放。否則模型分數的提升可能完全被交易摩擦吃掉。本階段只定義介面與待測項，不投入實作。

---

## 6. 落地順序（按 ROI / 風險）

| 順序 | 動作 | 工程量 | 風險 | 對應原架構 |
|---|---|---|---|---|
| **1** | Kronos 訊號降噪：MC 平均報酬 + 用 Rank-IC 選 checkpoint | 小時級 | 低 | 元件一強化 |
| **2** | LightGBM **stacking** 層（橫斷面、rank 目標、Kronos + 技術 + 市場相對特徵） | 天級 | 低 | 元件三（重構 0.6/0.4） |
| **3** | Analog-RAG 當**特徵**接進 stacker（point-in-time 嚴格防洩漏） | 週級 | **高** | 元件二（重構 token-prefix） |

每一步都用 **Rank-IC** 量出增量貢獻；對不到增益就回滾。

---

## 7. 風險與紅線

1. **資料洩漏（最大殺手）**：重疊滑動窗、檢索庫非 point-in-time、stacker 非 out-of-fold。任一處出錯 → 回測漂亮但全假。
2. **tokenizer 過擬合**：資料 / regime 不足時先別動 tokenizer。
3. **單樣本訊號雜訊**：務必 MC 平均。
4. **backtest ≠ production**：成本 / 滑價未計前，分數不可當真。
5. **算力限制**：T4 上優先 small / predictor-only。

---

## 8. 成功指標與收斂

開發期先看 **Rank-IC** 趨勢；最終驗收對齊 `CLAUDE.md` 的收斂表：

| 指標 | 目標 |
|---|---|
| Sharpe Ratio | ≥ 1.5（vs `^TWII`） |
| Annual Return | > 15%（測試集 2024-07-01 起） |
| Max Drawdown | < 20% |
| 連續改善幅度 | < 1%（連續 2 輪則停止） |

---

## 9. 開放問題（待資料下載完、有 baseline 後決定）

- baseline 數字出來前不調參。
- 瓶頸在**排序品質**還是**風控**？決定 1 / 2 / 3 各投入多少。
- Layer 1 微調到哪一級（predictor-only vs full）能在 T4 預算內拿到最佳 Rank-IC？
- Layer 2 的檢索 embedding 用 Kronos encoder hidden state 還是輕量 hand-crafted 向量，何者增量貢獻高？
