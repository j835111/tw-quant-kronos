# Research Findings — Kronos TW 模型改進

**研究日期：** 2026-06-26  
**迭代次數：** 15  
**覆蓋類別：** 5/5  
**狀態：** BOUNDED (15 iterations)

---

## ICP Challenges（目標對象痛點）

### F1: 訓練目標與部署目標不對齊 ★★★ HIGH
**Problem:** Model trained on close-price token CE loss, but deployed for open-to-open execution.  
**Evidence:**
- 我們自己的實驗：v1（close signal）→ v2（open signal）Sharpe 從 1.196 → 1.356（+13.4%）
- v1 IC validation 在 `ic_validation.py` 用 `pred["close"][h]/ctx_close - 1` vs actual close，與執行目標不匹配
- *The Label Horizon Paradox* (arxiv:2602.03395, ICML 2026)：「訓練標籤不應直接等於推理目標；最優監督信號通常在中間 horizon，由動態 signal-noise 競爭決定。」
- 我們的 Round 0 fine-tuning 用 token CE 反而讓 val_loss 從 pretrained 2.9966 上升到 3.6440，IC 從 0.050 降到 0.041

### F2: 高-低振幅預測未被使用，而它可直接改善 MaxDD ★★★ HIGH
**Problem:** 模型輸出 `high` 和 `low` 但被完全忽略；MaxDD 35% 超過目標 20%。  
**Evidence:**
- 我們的模型輸出：`high-low/close ≈ 1.3-2.4%`（2330.TW 範例）
- [Forecasting and Trading the High-Low Range of Stocks (Springer)]：神經網路可有效預測日內 high/low，用於 long-short 策略
- [ATR-based position sizing studies]：`Position Size = Account Risk / (ATR × Multiple)` 為業界標準
- 2025 backtesting：ATR 結合方向指標改善策略獲利 34%
- [Do Better Volatility Forecasts Lead to Better Portfolios? (arxiv:2605.19278)]：GNN 預測波動率後用於 min-variance 優化，103 週測試（2024-2025）顯著改善

### F3: IC early stopping 噪音過高導致 epoch 選擇不穩 ★★ MEDIUM
**Problem:** 300 symbols × 20 dates = 只有 6000 點用來計算 IC-IR；Round 2 選了 epoch 4（IC-IR 0.4066）但表現比 Round 0 差。  
**Evidence:**
- 我們的 Round 2 失敗分析：IC-IR 噪音太高，選到的 epoch 不是真正最佳
- IC 在少量 dates 下方差很大（IC-IR 計算需要至少 20+ dates 才穩定）

### F4: 模型在 3d 和 10d 尺度 open 預測噪音高，只有 5d 有效 ★★★ HIGH
**Problem:** hold=3d Sharpe 0.05，hold=10d Sharpe 0.445，只有 hold=5d Sharpe 1.356 可用。  
**Evidence:** 我們的 v2 實驗數據

---

## Competitor Gaps（競爭者差距）

### F5: Competing models use continuous regression, not discrete tokenization ★★ MEDIUM
**Problem:** TimesFM、Chronos-2、Moirai 2.0 都用連續回歸；Kronos 的 BSQ 離散化會引入量化誤差。  
**Evidence:**
- [Re(Visiting) Time Series Foundation Models in Finance (arxiv:2511.18578)]
- [Chronos-2: From Univariate to Universal Forecasting (arxiv:2510.15821)]：Moirai 2.0 從 encoder 換成 decoder-only + quantile head
- TimesFM：連續 latent space，supervised regression loss

### F6: Ranking/listwise 損失函數顯著優於點估計損失，但 Kronos 只用 token CE ★★★ HIGH
**Problem:** 股票排名任務需要 pairwise/listwise 損失，但我們用的 token cross-entropy 本質是點估計損失（不直接優化相對排名）。  
**Evidence:**
- [On Evaluating Loss Functions for Stock Ranking (arxiv:2510.14156, CIKM 2025)]：系統性比較 pointwise/pairwise/listwise 損失函數，ranking 損失明顯更優
- [MiM-StocR (arxiv:2509.10461)]：Adaptive-k ApproxNDCG (listwise) + 動能多任務 MTL → 在 CSI 50/100/300 上 SOTA
- 研究一致表明：為排名優化的損失函數在股票選擇中 > 回歸損失

---

## Market Trends（市場趨勢）

### F7: The Label Horizon Paradox — 最優訓練標籤 ≠ 推理目標 ★★★ HIGH
**Problem / Finding:** 最優代理標籤在中間 horizon，不是直接對應部署目標。  
**Evidence:**
- [The Label Horizon Paradox (arxiv:2602.03395, ICML 2026)]：bi-level optimization 自動找最優 proxy label；在訓練時解耦標籤與推理目標能一致改善所有架構
- 實驗設置與我們接近：chronological split，2019-2023 train，2023-2024 val，2024-2025 test

### F8: Volume 預測 + 不確定性過濾可大幅提升動能策略 ★★ MEDIUM
**Problem / Finding:** 結合開盤報酬信號與 volume-based 不確定性，在高不確定性 regime 達 71.43% 準確度，Sharpe 3.02。  
**Evidence:**
- [Enhancing Intraday Momentum Prediction via Volume-Based Information Uncertainty (MDPI 2025)]：opening returns + volume uncertainty = 顯著改善
- [Forecasting Intraday Volume in Equity Markets with ML (arxiv:2505.08180)]：惜日內成交量高度可預測

### F9: Lookback window 過長可能反而降低 Transformer 預測力 ★ LOW
**Evidence:**
- [Overcoming Lookback Window Limitations (OpenReview)]：過長窗口引入冗餘 → attention 分散
- 但其他架構（xLSTM-style）較長窗口更好
- 我們目前 lookback_window=90，對 Transformer 可能偏長
- 信心低因為證據相互矛盾

---

## UX & Experience（模型輸出利用）

### F10: Multi-task 輔助目標（high/low/volume）可改善主任務精度 ★★ MEDIUM
**Finding:** 把 high/low/volume 作為輔助預測目標（即使主信號只用 open/close）可提升模型整體表現。  
**Evidence:**
- [Autoencoder-based Hybrid Multi-Task Predictor for OHLC Prices (arxiv:2204.13422)]：OHLC 多任務預測互相強化
- [MiM-StocR]：動能線指標作為輔助任務改善股票排名
- Kronos 已有 6 欄輸出，loss 只算 token CE；可以加 auxiliary MSE on price space

### F11: 預測 ATR = 前向風險管理工具（不需要歷史 ATR） ★★★ HIGH
**Finding:** 用模型預測的 `(high - low)` 代替歷史 ATR 做 position sizing，可以在進場前就知道預期波動。  
**Mechanism:** `pos_size_i ∝ 1 / pred_ATR_i`，在 `signals_to_holdings()` 或 portfolio construction 時套用  
**Evidence:** F2 同源
