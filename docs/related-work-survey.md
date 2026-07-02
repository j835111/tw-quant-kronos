# 相關專案與論文調研：金融時序基礎模型

> 調研日期：2026-07-01  
> 調研範圍：GitHub、Hugging Face、arXiv  
> 目標：找出與 Kronos（OHLCV Transformer 基礎模型）主題相似的專案，整理可借鑑的技術做法

---

## 摘要

本次調研共收集 15 個來源，重點分析 6 個專案/論文。核心結論：

1. **領域原生預訓練是必要的**——通用時序基礎模型零樣本遷移到金融幾乎無效
2. **Tokenization 設計有多種成熟方案**可借鑑（patch、VQ-VAE、mean scaling + binning）
3. **Ranking loss 優於 MSE**，但預測品質指標（IC）與投組績效（Sharpe）是脫鉤的
4. **Codebook 多樣性保護**是 VQ-based 模型的關鍵，目前 Kronos 的 BSQuantizer 缺乏此機制

---

## 1. STORM — 雙模組 VQ-VAE（時序 × 截面）

**來源：** arXiv 2412.09468

### 架構特點

- **雙 VQ-VAE**：分別處理時序維度（單支股票 × p 天）與截面維度（所有股票 × 單一交易日），再以 cross-attention 融合
- **時序 patch（TS）**：`[1 stock × p days]` 為一個 token
- **截面 patch（CS）**：`[all stocks × 1 day]` 為一個 token

### 訓練損失設計

```
L_total = L_recon + L_quantize + λ₁·L_diversity + λ₂·L_orthogonality
```

- `L_diversity`：讓不同 codebook entry 被均勻使用，避免 mode collapse
- `L_orthogonality`：讓 codebook 向量互相正交，提升表達能力
- **不包含 ranking loss**；Rank IC、ICIR 僅作為事後評估指標，不進入梯度更新

### 對 Kronos 的啟示

| 問題 | STORM 的做法 |
|------|-------------|
| Codebook collapse（少數 entry 被過度使用）| diversity loss |
| Codebook 表達能力不足 | orthogonality loss |
| 多資產截面資訊利用 | 截面 patch + cross-attention |

**建議行動**：在 BSQuantizer 訓練損失中加入 diversity + orthogonality 項。

---

## 2. FinCast — 10 億參數金融時序基礎模型

**來源：** arXiv 2508.19609

### 架構特點

- **Decoder-only Transformer**，10 億參數
- **Patch-based tokenization**：輸入序列切成長度 P 的非重疊 patch，而非逐步預測
- **Instance normalization**：每個輸入樣本獨立標準化，消除跨品種量級偏差，使模型可跨品種零樣本泛化
- **Sparse MoE**：4 experts，top-k=2 routing，token 級稀疏激活
- **Learnable frequency embeddings**：編碼不同時間解析度（日/週/月），讓單一模型處理多頻率

### 訓練損失設計

**Point-Quantile Loss（PQ-loss）**，由三項組合：

```
L_PQ = L_huber + L_quantile + L_trend
```

- `L_huber`：點預測，對 outlier 較穩健
- `L_quantile`：多分位數回歸，輸出預測分佈
- `L_trend`：trend consistency，讓預測方向一致

聲稱比純 MSE 更能應對金融序列的 fat tail 與非穩態。

### 對 Kronos 的啟示

- PQ-loss 可替換現有的純 MSE 預測損失，提升對極端行情的穩健性
- Instance normalization 比 per-window z-score 更能處理跨股票的量級差異

---

## 3. Ranking Loss 實證比較

**來源：** arXiv 2510.14156

### 實驗設定

- 資料：S&P 500 日線 OHLCV
- 模型：Transformer
- 比較：MSE、RankNet、ListNet、Margin ranking loss 等

### 主要結果

| Loss | 年化報酬 | Sharpe | IC Spearman |
|------|---------|--------|-------------|
| Margin ranking | **16.23%** | **0.75** | — |
| ListNet | 16.00% | 0.74 | — |
| MSE（baseline） | 14.78% | 0.66 | — |
| RankNet | — | — | **最高（0.077）** |

### 關鍵發現

> **IC 最高的 loss 不等於最佳投組績效。** RankNet 的 IC Spearman 最高，但年化報酬和 Sharpe 都不是最優。

這與 Kronos 專案的觀察完全一致（見 `forecast_eval.md`）：預測品質指標與投組績效是脫鉤的。

**建議行動**：以 Margin ranking loss 作為下一輪訓練的主要損失函數實驗對象。

---

## 4. 通用時序基礎模型在金融的邊界

**來源：** arXiv 2511.18578、arXiv 2606.27100

### 零樣本表現（直接用預訓練 TSFM）

| 模型 | R²（金融報酬預測） | 方向準確率 |
|------|-----------------|-----------|
| Chronos large | −1.37% | 51.x% |
| TimesFM 500M | −2.80% | ~50% |
| CatBoost（baseline） | −0.10% | — |

結論：**零樣本遷移幾乎無效**，甚至輸給傳統 ML baseline。

### 從頭在金融資料預訓練

Chronos small 從頭在金融資料訓練後：年化報酬 36.84%，Sharpe 5.42（vs 零樣本 R²=−1.27%）。

### 對 Kronos 的啟示

> **通用時序預訓練不遷移到金融，領域原生預訓練是必要的。**

這直接驗證了 Kronos 的核心設計決策（在金融 K 線資料上從頭預訓練）是正確的方向。微調時也應優先使用 Kronos 自身的預訓練權重，而非嘗試遷移其他通用時序模型。

---

## 5. Chronos Tokenization 策略

**來源：** arXiv 2507.07296（Amazon Chronos）

### 核心設計

```
原始時序值
  → mean scaling（每個 context window 做均值正規化）
  → uniform binning（均勻量化到 K 個 bin）
  → 離散 token
  → categorical cross-entropy loss 訓練
```

這個設計把**迴歸問題轉化為分類問題**，與 Kronos 的 BSQuantizer 方向相同，但實作不同：

| | Chronos | Kronos |
|---|---------|--------|
| 量化方式 | Uniform binning | Binary Spherical Quantization |
| 訓練目標 | Categorical CE | Reconstruction + VQ |
| 正規化 | Mean scaling | Per-window z-score |

---

## 6. 工具框架

### allRank（allegro/allRank）

實作 9 種排序損失函數，涵蓋 pointwise、pairwise、listwise 三大類：

- **Listwise**：LambdaRank、LambdaLoss、ApproxNDCG、NeuralNDCG
- **Pairwise**：RankNet、Margin ranking
- **Pointwise**：MSE

可直接整合進 Kronos 訓練流程，快速實驗不同 ranking loss。

### Microsoft Qlib

- **Rank IC / ICIR** 是業界標準的截面排名信號評估指標
- 建議將 Rank IC 納入 Kronos 每 epoch 的評估，作為 early stopping 的補充依據（與現有的 val_loss 並行）

---

## 行動優先順序

| 優先 | 來源 | 建議行動 | 預期效益 |
|------|------|---------|---------|
| ⭐⭐⭐ | STORM | BSQuantizer 加入 diversity + orthogonality loss | 防 codebook collapse，提升表達能力 |
| ⭐⭐⭐ | arXiv 2510.14156 | 試 Margin ranking loss 替換現有 ranking loss | 文獻中最佳投組表現 |
| ⭐⭐ | FinCast | PQ-loss（Huber + 分位數 + trend）替換純 MSE | 更穩健的預測損失 |
| ⭐⭐ | Qlib | 加入 Rank IC 作為 early stopping 指標 | 更直接對齊投組目標 |
| ⭐ | FinCast | Instance normalization 跨股票 scale 正規化 | 改善跨股票泛化 |

---

## 參考來源

1. [STORM: Spatio-Temporal Stock Return Foundation Model](https://arxiv.org/abs/2412.09468)
2. [FinCast: A Large-Scale Financial Time Series Foundation Model](https://arxiv.org/abs/2508.19609)
3. [Ranking Loss for Financial Time Series Forecasting](https://arxiv.org/abs/2510.14156)
4. [TSFMs for Financial Return Forecasting](https://arxiv.org/abs/2511.18578)
5. [Benchmarking TSFMs on Financial Tasks](https://arxiv.org/abs/2606.27100)
6. [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2507.07296)
7. [allRank: Reproducible Ranking with PyTorch](https://github.com/allegro/allRank)
8. [Microsoft Qlib](https://github.com/microsoft/qlib)
9. [Moirai 2.0](https://huggingface.co/Salesforce/moirai-2.0-R-small)
