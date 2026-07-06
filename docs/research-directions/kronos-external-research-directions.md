# Kronos 外部調研與後續方向

**日期：** 2026-07-01  
**調研範圍：** GitHub、Hugging Face、arXiv  
**目的：** 盤點與 `Kronos` 最接近的研究與開源路線，整理成後續可執行方向。

---

## 先講結論

`Kronos` 現在最有價值的差異化，不是變成另一個泛用 time-series foundation model，而是把「**金融專用 tokenization + 金融專用 benchmark + 多資產 / 多模態能力**」做成完整體系。

若只選一條最優先路線，建議先做：

1. **Kronos-2 for Finance**：支援多資產、multivariate、covariates
2. **金融 benchmark 套件**：固定 rolling protocol + 經濟指標評估
3. **representation / embedding 路線**：讓 Kronos 不只做 autoregressive forecasting，也能做 ranking、retrieval、stacking

---

## Kronos 目前位置

依 `NeoQuasar/Kronos-base` 的模型卡，Kronos 的核心定位是：

- 金融市場專用 foundation model，而非泛用 TSFM
- 兩階段架構：`OHLCV -> hierarchical tokens -> autoregressive Transformer`
- 以 45+ 交易所、超過 120 億筆 K-line 預訓練
- 現有公開模型 context 以 512 為主（`Kronos-small` / `Kronos-base`）

這個定位本身是正確的，因為近年的外部文獻也逐漸指出：**金融資料的高噪音、non-stationarity、跨市場異質性，讓「通用 TSFM 直接套 finance」的效果有限，領域專用預訓練仍然重要。**

---

## 外部 landscape

### A. 通用 TSFM 主線

#### 1. Chronos / Chronos-2

- `Chronos` 的核心做法是把 time series 量化成 token，再用語言模型式訓練；這和 `Kronos` 的 tokenization 思路最接近。
- `Chronos-2` 已把能力推到：
  - univariate
  - multivariate
  - covariate-informed forecasting
  - 更長 context（模型卡列出 max context 8192）
- 它代表目前最清楚的升級方向：**從單序列預測走向 universal forecasting**。

**對 Kronos 的啟示：**
- 不該只停在單資產 OHLCV 預測
- 下一代版本應該原生支援：
  - cross-asset context
  - market covariates
  - future-known covariates

#### 2. TimesFM

- `TimesFM` 是 decoder-only 路線，和 `Kronos` 在架構取向上更接近。
- 新版公開 checkpoint 已走向更長 context、直接 forecasting API、較成熟的推論介面。
- 它的重點不是金融，而是證明 **decoder-only TSFM** 這條路本身可行。

**對 Kronos 的啟示：**
- `Kronos` 的 decoder-only 主幹不需要推翻
- 更值得補的是資料接口、長 context、batch / multivariate 推論能力

#### 3. MOMENT

- `MOMENT` 強調的不只是 forecasting，而是通用時間序列表示學習：
  - forecasting
  - classification
  - anomaly detection
  - imputation
  - embedding
- 它的價值在於把 TSFM 變成「可重用表徵骨幹」而不只是預測器。

**對 Kronos 的啟示：**
- `Kronos` 應該補 `embedding / representation` 使用方式
- 金融實務中，ranking、retrieval、stacking 常比直接 price forecast 更有 alpha 價值

#### 4. Time-MoE / Timer-S1

- 這條線重點在 scaling：
  - 更大模型
  - 更大資料集
  - 更長 context
  - mixture-of-experts
- `Time-MoE` repo 甚至直接把「covariate support」列為待辦，代表這仍是高速演進中的能力。

**對 Kronos 的啟示：**
- 規模化很重要，但不是第一優先
- 在金融領域，先把任務對齊與資料對齊做好，通常比先盲目擴參數更重要

### B. 金融專用研究主線

#### 1. FinCast

- `FinCast` 自稱是 financial time-series forecasting foundation model，重點在：
  - temporal non-stationarity
  - multi-domain diversity
  - varying temporal resolutions
  - zero-shot robustness

**對 Kronos 的啟示：**
- 金融專用 foundation model 已開始出現競品
- `Kronos` 需要更明確定義自己的 moat：
  - candlestick-native tokenization
  - cross-market pretraining
  - downstream trading-oriented evaluation

#### 2. Re(Visiting) Time Series Foundation Models in Finance

- 這篇最重要的訊息不是新架構，而是實證結果：
  - off-the-shelf TSFM 在金融 zero-shot / fine-tune 不一定好
  - 從金融資料 scratch pretrain 反而顯著更強
  - synthetic augmentation、dataset size、hyperparameter tuning 都有幫助

**對 Kronos 的啟示：**
- 這篇基本上在替 `Kronos` 的大方向背書
- 但也說明光有模型不夠，**資料規模、資料清洗、評估 protocol** 同樣是核心資產

#### 3. FinMultiTime

- 這是 multimodal 金融資料集方向：
  - 新聞
  - 財報表格
  - K-line 技術圖
  - 價格時間序列
- 金融模型正在從純數值序列走向多模態對齊。

**對 Kronos 的啟示：**
- `Kronos` 若只留在 OHLCV，長期差異化不夠
- 中期應考慮把：
  - news
  - fundamentals
  - market regime metadata
  和 K-line token 一起建模

#### 4. MarketGPT

- 這條線不是做 OHLCV forecasting，而是往 market microstructure / order flow / LOB 模擬前進。
- 難度高很多，但辨識度也最高。

**對 Kronos 的啟示：**
- 若未來要做高風險高報酬研究支線，可往：
  - event-level tokenization
  - order-flow generation
  - simulator-driven pretraining
  前進

---

## 從這些文獻得到的五個判斷

### 1. 金融領域專用預訓練仍然必要

外部研究沒有推翻 `Kronos` 的核心前提，反而更支持它：**finance 不是把通用 TSFM 拿來套就好。**

對應策略：

- 保持金融專用預訓練主線
- 進一步擴大市場覆蓋、頻率覆蓋、regime 覆蓋
- 把 synthetic augmentation 納入正式訓練管線

### 2. 多資產 / 多變量 / covariates 是下一代基線能力

`Chronos-2` 已把這件事做成公開基線。如果 `Kronos` 下一步還停在單序列自回歸，會開始落後。

對應策略：

- 支援多檔標的共同輸入
- 支援 target + covariates 分組建模
- 支援 market-wide context，例如：
  - index
  - sector ETF
  - rates / FX / crypto proxy
  - calendar features

### 3. 只看 forecast loss 不夠，必須用經濟指標定義成敗

金融上最重要的是：

- ranking quality
- portfolio return
- Sharpe
- Max drawdown
- turnover / cost-adjusted PnL

不是 token loss 或 point forecast RMSE 本身。

對應策略：

- 將 benchmark 正式化
- 預設產出經濟指標，不只預測誤差
- 明確區分：
  - forecast benchmark
  - signal benchmark
  - portfolio benchmark

### 4. representation learning 可能比直接 forecasting 更有產值

`MOMENT` 給的最大啟發是：TSFM 可以是 backbone。  
對金融來說，`Kronos embedding -> ranking / retrieval / stacker` 可能比端到端價格生成更穩。

對應策略：

- 釋出 hidden-state / embedding API
- 做 retrieval、clustering、regime detection
- 做 downstream ranker（LightGBM / XGBoost / MLP / LambdaRank）

### 5. 多模態是中期 moat，微結構是長期 moonshot

若只做數值 OHLCV，Kronos 的競爭力會主要停在「金融版 Chronos」。  
若加上多模態與微結構，才有機會變成真正難取代的研究線。

---

## 建議的後續方向

### P0: 最近 2-4 週內就該做

#### 方向 1：建立金融 benchmark 套件

先不要急著加更大模型。先把「怎樣算變好」這件事做硬。

至少應固定：

- rolling / walk-forward protocol
- train / val / test cut
- close-close 與 open-open 的一致定義
- transaction cost / slippage 假設
- ranking metrics：IC、Rank IC、IC-IR
- portfolio metrics：Ann、Sharpe、MaxDD、turnover

**原因：**
- 沒有 benchmark，後面的每個模型改動都容易重複犯錯
- 這也最符合目前 `finetune_tw/` 既有工作流

#### 方向 2：把 Kronos 做成「可抽 embedding 的 backbone」

在現有 autoregressive 預測外，新增：

- sequence embedding
- window embedding
- per-step hidden state export

然後先接一層簡單 downstream：

- cross-sectional ranker
- retrieval-based analog search
- stacking model

**原因：**
- 和現有 codebase 相容
- 研發風險比重訓大型模型低
- 很適合先在 `finetune_tw/` 驗證實際 alpha

### P1: 下一個 1-2 月

#### 方向 3：Kronos-2 for Finance

核心目標：把 `Kronos` 從金融單序列 token model 升級成金融 universal forecasting model。

最低限度應支援：

- multivariate OHLCV
- cross-asset grouped input
- historical covariates
- known-future covariates
- context length 擴展

**建議原則：**
- 保留金融 tokenization 的核心優勢
- 不必照抄 `Chronos-2` 架構
- 但任務能力至少要對齊同一世代基線

#### 方向 4：正式納入 synthetic data 與 regime-aware pretraining

可做的資料增強方向：

- bull / bear / sideways regime rebalancing
- volatility shock augmentation
- cross-market resampling
- multi-frequency mixing

**原因：**
- `Chronos`、`Chronos-2`、`Re(Visiting...)` 都指出 synthetic / scale 對泛化有幫助
- 對金融尤其重要，因為極端 regime 在真實資料中本來就稀少

### P2: 下一季

#### 方向 5：multimodal Kronos

把以下訊號與價格對齊：

- 新聞文本
- 財報表格 / fundamentals
- K-line 圖像
- macro / event metadata

優先順序建議：

1. metadata / calendar / fundamentals
2. news text
3. chart image

**原因：**
- 表格與 metadata 的 integration 成本通常比影像低
- 能更快驗證是否有增益

#### 方向 6：microstructure / order-flow 支線

這條線適合獨立成 research branch：

- event-level tokenizer
- LOB / order message modeling
- synthetic market simulation

**原因：**
- 研究辨識度很高
- 但和現有 daily / candlestick pipeline 差距大，不適合和主線混做

---

## 對本 repo 的具體落點

### 1. `model/`

建議增補：

- `KronosPredictor` 的 embedding / hidden-state export
- grouped / multivariate input 介面
- covariate-aware inference API

### 2. `finetune_tw/`

最適合先落地：

- 金融 benchmark 固化
- ranking-oriented eval
- embedding -> ranker pipeline
- analog retrieval baseline

### 3. `tests/`

要補的不是只有 unit test，還包括：

- deterministic eval slices
- benchmark regression tests
- backtest schema / metrics regression

### 4. `docs/`

建議後續再補兩份：

- `benchmark-spec`：明確定義金融評估 protocol
- `kronos-2-finance-roadmap`：將 P0 / P1 / P2 拆成可執行任務

---

## 我建議的優先順序

若只能排三件事：

1. **先把 benchmark 做硬**
2. **再把 Kronos 變成可抽表徵的 backbone**
3. **最後再做 multivariate + covariates 的 Kronos-2 for Finance**

原因很直接：

- benchmark 決定你有沒有正確優化目標
- representation 路線最接近現有 repo，風險最低
- multivariate / covariate 升級是必要，但工程量與研究不確定性都更高

---

## 開放問題

後續立項前，至少要先釐清：

1. `Kronos` 的主要任務要定義成 forecasting、ranking，還是兩者並行？
2. 下一代模型要優先追求：
   - 更好的 zero-shot？
   - 更好的 fine-tune？
   - 更好的回測表現？
3. 金融 covariates 的最小可行集合是什麼？
4. `embedding -> ranker` 是否已足以超過直接自回歸 fine-tune？
5. multimodal 第一個模態要接 fundamentals 還是 news？

---

## 參考來源

### GitHub

- Chronos: https://github.com/amazon-science/chronos-forecasting
- TimesFM: https://github.com/google-research/timesfm
- MOMENT: https://github.com/moment-timeseries-foundation-model/moment
- Time-MoE: https://github.com/Time-MoE/Time-MoE

### Hugging Face

- Kronos-base: https://huggingface.co/NeoQuasar/Kronos-base
- Chronos-2: https://huggingface.co/amazon/chronos-2
- TimesFM 2.5 200M: https://huggingface.co/google/timesfm-2.5-200m-pytorch
- MOMENT-1-large: https://huggingface.co/AutonLab/MOMENT-1-large

### arXiv / 論文

- Chronos: https://arxiv.org/abs/2403.07815
- Chronos-2: https://arxiv.org/abs/2510.15821
- TimesFM: https://arxiv.org/abs/2310.10688
- MOMENT: https://arxiv.org/abs/2402.03885
- Time-MoE: https://arxiv.org/abs/2409.16040
- Timer-S1: https://arxiv.org/abs/2603.04791
- FinCast: https://arxiv.org/abs/2508.19609
- Re(Visiting) Time Series Foundation Models in Finance: https://arxiv.org/abs/2511.18578
- FinMultiTime: https://arxiv.org/abs/2506.05019
- MarketGPT: https://arxiv.org/abs/2411.16585

---

## 補充判斷

截至 2026-07-01，我沒有查到 `FinCast`、`FinMultiTime`、`MarketGPT` 對應到和 `Chronos` / `TimesFM` 同等成熟度的公開 GitHub / Hugging Face 生態，因此它們目前更適合當作**研究方向參照**，而不是直接拿來復用的工程基底。
