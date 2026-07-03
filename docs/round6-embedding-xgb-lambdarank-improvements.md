# Kronos TW Round 6 後續改進方案（XGBoost + LambdaRankIC）

**日期**：2026-07-02

**作者**：Antigravity

**定位**：Round 6 完成後的診斷、修正與後續實驗規劃

**目的**：針對 Round 6 (Kronos Embedding + XGBoost LambdaRankIC) 的回測失效問題（Sharpe 0.34，受困於 2026-Q2 極端動能行情），進行源碼診斷並提出具體的架構與特徵工程優化方案。

---

## 1. 核心代碼缺陷診斷 (Code Flaw Analysis)

### 🔴 缺陷一：時間序列特徵的「均值池化稀釋」 (Temporal Feature Dilution)
* **代碼位置**：[extract_embeddings.py:L26-30](https://github.com/j835111/Kronos/blob/6034e69255ade134d565034c499eb727d629c7aa/finetune_tw/extract_embeddings.py#L26-L30)
* **分析**：
  在 `extract_embeddings_batch` 中，默認對 Transformer 最後一層的隱藏狀態（hidden state）進行了 `context.mean(dim=1)`（對長度為 90 的時間維度求平均）。
  * **後果**：時間序列的「均值」會徹底消除價格隨時間變化的**順序與時序特徵**（如近期物極必反的暴跌 vs 連續上漲，其均值可能完全相同）。
  * **改進理據**：對於 Causal Decoder-only Transformer，最後一個 token 的隱藏狀態 `context[:, -1, :]` 通過自注意力（Self-Attention）機制，天然地聚合了整個 Lookback Window 的歷史資訊，且更側重於最新的狀態。只做均值池化，相當於主動丟棄了時序預測中最關鍵的「最新狀態」與「動能趨勢」。

### 🔴 缺陷二：特徵缺乏「橫截面相對排名」 (Lack of Cross-Sectional Relative Ranks)
* **代碼位置**：[extract_embeddings.py:L57-73](https://github.com/j835111/Kronos/blob/6034e69255ade134d565034c499eb727d629c7aa/finetune_tw/extract_embeddings.py#L57-L73)
* **分析**：
  雖然 XGBoost 是使用 `LambdaRankIC` 的**排名損失函數**進行訓練，但是輸入的技術指標（如 `feat_momentum_10 = 0.05`、`feat_ma20_dist = 0.02`）均為**絕對值**。
  * **後果**：在金融市場中，絕對值特徵的物理含義取決於市場整體環境（Regime）。在熊市中，10天報酬率 +5% 是極強的領先股；在狂牛市中，+5% 卻可能是嚴重的落後股。XGBoost 僅看絕對值，無法區分該股票在當前日期相對於全市場的排名強度。
  * **改進理據**：加入 **橫截面百分位數排名（Cross-Sectional Rank）** 特徵。例如將特徵轉化為 $0.0 \sim 1.0$ 的百分比排名，使樹模型能直觀識別「該股票在今日市場中處於前 5% 的強勢地位」，這與排名優化目標及 Top-K 策略高度一致，且天然具備 Regime 魯棒性。

### 🔴 缺陷三：特徵工程維度單一，缺乏多尺度與波動特徵
* **代碼位置**：[extract_embeddings.py:L57-73](https://github.com/j835111/Kronos/blob/6034e69255ade134d565034c499eb727d629c7aa/finetune_tw/extract_embeddings.py#L57-L73)
* **分析**：
  現有輔助技術特徵僅包含 `ma5_dist`、`ma20_dist`、`momentum_10` 和 `volume_ratio`。
  * **後果**：
    1. **動能維度單一**：僅有一個 10 天的動能特徵，無法讓模型識別「短期超跌反彈」（如 3D 暴跌但 20D 上漲）與「持續性趨勢」（如 3D, 10D, 20D 連續上漲）的區別。這也是 XGBoost 容易在動能狂飆期（如 2026-Q2）誤判為均值回歸的主因。
    2. **缺乏風險/波動率度量**：沒有包含波動率（Volatility）特徵，使得模型無法依據市場恐慌程度或個股波動度進行風險過濾。

---

## 2. 具體改進方案設計 (Proposed Architecture Improvements)

為了落實自主研調的結論，我們設計了以下三個具體的代碼級改進方案。

### 🛠️ 方案 A：引入 Last-Token / Concat 多模式池化策略
在 [extract_embeddings.py](https://github.com/j835111/Kronos/blob/6034e69255ade134d565034c499eb727d629c7aa/finetune_tw/extract_embeddings.py) 中，擴展池化方法，允許同時抓取「長期均值背景」與「短期最新狀態」：

```python
# 修改 extract_embeddings_batch 函數，增加 pooling 參數支持
def extract_embeddings_batch(
    predictor,
    df_list: list[pd.DataFrame],
    x_timestamp_list: list[pd.Series],
    layer_indices: list[int] | None = None,
    pooling: str = "concat",  # 可選 "mean", "last", "concat"
) -> np.ndarray:
    ...
    with torch.no_grad():
        s1_ids, s2_ids = predictor.tokenizer.encode(x_tensor, half=True)
        model = predictor.model

        if layer_indices is None:
            _, context = model.decode_s1(s1_ids, s2_ids, x_stamp_tensor)
            if pooling == "mean":
                pooled = context.mean(dim=1)
            elif pooling == "last":
                pooled = context[:, -1, :]
            elif pooling == "concat":
                # 同時拼接均值特徵與最新狀態特徵
                pooled = torch.cat([context.mean(dim=1), context[:, -1, :]], dim=-1)
```

### 🛠️ 方案 B：自動化特徵工程與橫截面相對強度（CS Ranks）
在 [extract_embeddings.py](https://github.com/j835111/Kronos/blob/6034e69255ade134d565034c499eb727d629c7aa/finetune_tw/extract_embeddings.py) 導出 Parquet 前，自動計算當日所有股票的橫截面相對排名，並重構 [train_xgb_lambdarank.py](https://github.com/j835111/Kronos/blob/6034e69255ade134d565034c499eb727d629c7aa/finetune_tw/train_xgb_lambdarank.py) 的特徵讀取邏輯，擺脫硬編碼限制。

1. **特徵橫截面排行 (Cross-Sectional Rank)**：
```python
# 在 build_embedding_dataset 保存 dataframe 前加入：
def build_embedding_dataset(cfg, predictor, symbols, rebal_dates, horizon):
    ...
    df = pd.DataFrame(rows)
    if not df.empty:
        # 找出所有技術特徵列
        tech_cols = [c for c in df.columns if c.startswith("feat_")]
        # 計算每日的橫截面百分比排名 (0 ~ 1)
        for col in tech_cols:
            df[f"{col}_cs_rank"] = df.groupby("date")[col].rank(pct=True)
    return df
```

2. **特徵列自動感知 (Feature Auto-Detection)**：
   修改 [train_xgb_lambdarank.py:L26-30](https://github.com/j835111/Kronos/blob/6034e69255ade134d565034c499eb727d629c7aa/finetune_tw/train_xgb_lambdarank.py#L26-L30)，自動載入所有以 `feat_` 開頭的特徵（包括原始值與新計算的 cs_rank），無需手動修改特徵列表：
```python
def _feature_columns(df: pd.DataFrame) -> list[str]:
    emb_cols = sorted([c for c in df.columns if c.startswith(EMBEDDING_PREFIX)],
                      key=lambda c: int(c[len(EMBEDDING_PREFIX):]))
    # 自動加載所有以 feat_ 開頭的技術特徵，包括 cs_rank
    tech_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    return emb_cols + tech_cols
```

### 🛠️ 方案 C：多尺度動能與波動率技術指標庫
重構 `compute_technical_features`，引入更全面的量價因子結構：

```python
def compute_technical_features(df: pd.DataFrame) -> dict[str, float]:
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    last_close = float(close[-1])

    # 1. 多尺度移動平均偏離度
    ma5 = float(close[-5:].mean()) if len(close) >= 5 else float(close.mean())
    ma20 = float(close[-20:].mean()) if len(close) >= 20 else float(close.mean())
    ma60 = float(close[-60:].mean()) if len(close) >= 60 else float(close.mean())

    # 2. 多尺度動能 (3天, 5天, 10天, 20天, 60天)
    mom_3 = float(last_close / close[-4] - 1.0) if len(close) > 3 and close[-4] != 0 else 0.0
    mom_5 = float(last_close / close[-6] - 1.0) if len(close) > 5 and close[-6] != 0 else 0.0
    mom_10 = float(last_close / close[-11] - 1.0) if len(close) > 10 and close[-11] != 0 else 0.0
    mom_20 = float(last_close / close[-21] - 1.0) if len(close) > 21 and close[-21] != 0 else 0.0
    mom_60 = float(last_close / close[-61] - 1.0) if len(close) > 61 and close[-61] != 0 else 0.0

    # 3. 波動度度量 (10日與30日日收益率標準差)
    returns = close[1:] / close[:-1] - 1.0
    vol_10 = float(returns[-10:].std()) if len(returns) >= 10 else 0.0
    vol_30 = float(returns[-30:].std()) if len(returns) >= 30 else 0.0

    # 4. 量比與量能趨勢
    recent_vol_mean = float(volume[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
    vol_ratio = float(volume[-1] / recent_vol_mean) if recent_vol_mean != 0 else 1.0
    vol_trend = float(volume[-5:].mean() / volume[-20:].mean()) if len(volume) >= 20 and volume[-20:].mean() != 0 else 1.0

    # 5. 日內價差震盪幅度
    high_low = (df["high"] - df["low"]) / df["close"]
    hl_spread_5 = float(high_low.iloc[-5:].mean())

    return {
        "feat_ma5_dist": float(last_close / ma5 - 1.0) if ma5 != 0 else 0.0,
        "feat_ma20_dist": float(last_close / ma20 - 1.0) if ma20 != 0 else 0.0,
        "feat_ma60_dist": float(last_close / ma60 - 1.0) if ma60 != 0 else 0.0,
        "feat_momentum_3": mom_3,
        "feat_momentum_5": mom_5,
        "feat_momentum_10": mom_10,
        "feat_momentum_20": mom_20,
        "feat_momentum_60": mom_60,
        "feat_vol_10": vol_10,
        "feat_vol_30": vol_30,
        "feat_volume_ratio": vol_ratio,
        "feat_volume_trend": vol_trend,
        "feat_hl_spread_5": hl_spread_5,
    }
```

---

## 3. 原方案補充評估（Claude，2026-07-03）

> 以下為對上述方案的獨立評估，原文未做任何修改。三個代碼診斷已對照 `research/round-6-m1-embedding` 分支原始碼逐一核對。

## 診斷核實結果

| 診斷 | 核實 | 備注 |
|------|------|------|
| 缺陷一：mean pooling 稀釋時序 | ✅ 屬實 | `extract_embeddings.py` 的 `decode_s1` 路徑確為 `context.mean(dim=1)`。**補充：`layer_indices` 消融路徑也是 mean-pool，方案 A 應一併修改** |
| 缺陷二：缺橫截面排名 | ✅ 屬實 | 4 個技術特徵全是絕對值。三個診斷中最有價值——cs_rank 直擊 Round 6 事後分析的病因（regime 依賴），與 LambdaRankIC 目標對齊，同日截面排名無前視洩漏 |
| 缺陷三：動能維度單一 | ✅ 屬實 | 與 round-history 的修正方向「加動能特徵」完全一致；`_TECH_FEATURE_COLUMNS` 確為硬編碼 4 特徵 |

## 兩個問題

### 問題 1：重大遺漏——未處理驗證期單一 regime

三個方案全在**特徵側**，但 early stopping 仍用 2024H1 單一窗口選 best_iteration。Round 6 事後分析（`docs/kronos-tw-round-history.md` Round 6 章節）指出這正是把模型鎖進反轉 regime 的機制之一——**如果 2024H1 行情獎勵反轉，新加的動能特徵很可能在樹分裂中被忽略，特徵給了也學不到**。多 regime 驗證期應與方案 B/C 同批實施。

文件也未涵蓋 round-history 已列的另外兩個方向：測試期逐期 IC 診斷（最便宜的假說確認實驗）與 Kronos+XGBoost ensemble（兩訊號相關性 0.678、性質互補）。本文件與這兩個方向互補而非衝突。

### 問題 2：成本排序未交代

- **方案 A（改 pooling）需要重跑整個 GPU embedding 抽取**——Round 6 最貴的一步（A40 多進程平行數小時），且 pod 目前 EXITED、embedding 檔案尚待恢復。concat 使 embedding 從 832 維翻倍到 1664，新舊 parquet 不可混用，必須全量重抽。
- **方案 B+C 完全不需要 Kronos**：技術特徵和 cs_rank 從本地 `tw_stocks.db` + CPU 就能算，可直接 merge 到既有 embeddings parquet 重訓 XGBoost。

**建議執行順序：先做逐期 IC 診斷確認 regime 假說 → B + C + 多 regime 驗證期（CPU 級成本）→ 證明有效後再投資 A（GPU 重抽）。** B+C 先行也順便回答懸置的「raw-feature-only 對照組」問題。

## 代碼細節（小瑕疵，不影響方向）

1. `mom_20` 的條件 `len(close) > 21` 差一（`close[-21]` 只需 `len >= 21`），無害但不精確；`mom_60` 同樣模式。
2. `hl_spread_5` 沒有 `len < 5` 或 `close = 0` 的防護（`iloc[-5:]` 短窗仍能算，風險低）。
3. 方案 B 的 `groupby("date").rank(pct=True)` 隱含假設「同一日的全部股票在同一個 parquet chunk 裡」——Round 6 的平行抽取按日期切段，此假設成立，但值得在代碼註明。
4. 方案 A 對 causal decoder 用 last-token 的理據正確，但 mean-pool 並非全無價值（last token 可能較噪），`concat` 同時保留兩者是穩妥選擇。

---

## 4. Parquet / XGBoost Artifact 實測（2026-07-03）

本節使用 Round 6 實際產物進行一次性驗證：

- `embeddings_train.parquet`：2,141,404 rows，2015-05-22 至 2023-12-29
- `embeddings_val.parquet`：135,323 rows，2024-01-01 至 2024-06-28
- `xgb_round6.json`：836 features，200 boosted rounds，best iteration 190

### 4.1 驗證指標可重現

| Trees | Mean rank-IC | IC-IR | Positive-IC dates | Mean top-10 excess |
|---|---:|---:|---:|---:|
| 全部 200 trees | 0.066281 | 0.636 | 70.0% | +0.216% |
| Best iteration 0-190 | 0.066450 | 0.637 | 70.8% | +0.213% |

190 與 200 trees 的結果幾乎相同，因此 boost rounds 與 early stopping
不是 Round 6 失敗的主要原因。

### 4.2 交易日曆缺陷

`extract_embeddings.py` 使用 `pd.bdate_range`，而非台股實際交易日：

- Train：147 / 2,246 dates 不是 TWSE 交易日（6.5%）
- Validation：13 / 130 dates 不是 TWSE 交易日（10.0%）
- Validation 有 12 天的 score state 與前一交易日完全相同
- 2024 農曆年附近的 2024-02-05 context 被重複計權八次

過濾後 validation 剩 117 個真實交易日：

| Trees | Mean rank-IC | IC-IR | Positive-IC dates | Mean top-10 excess |
|---|---:|---:|---:|---:|
| 全部 200 trees | 0.072783 | 0.705 | 72.6% | +0.300% |
| Best iteration 0-190 | 0.072909 | 0.704 | 73.5% | +0.293% |

此缺陷沒有灌高本次 validation IC；但它改變 train/validation 的樣本權重，
後續所有實驗必須先修正，否則不同版本無法公平比較。

### 4.3 模型確實偏反轉

以下數字已過濾非交易日：

| Feature | Train label IC | Validation score IC | Validation top-10 percentile | XGB total-gain share |
|---|---:|---:|---:|---:|
| MA5 distance | -0.0417 | -0.3415 | 36.5% | 20.6% |
| MA20 distance | -0.0291 | -0.1963 | 45.1% | 9.2% |
| 10-day momentum | -0.0218 | -0.1471 | 44.7% | 3.9% |
| Volume ratio | -0.0009 | -0.0504 | 60.0% | 4.0% |

四個 raw features 合計占 XGBoost total gain 的 37.6%，且是 gain 最高的四個
單一特徵。最重要的 MA5 distance 明確把模型推向近期落後股。因此「模型偏反轉」
已同時得到歷史 label 與 fitted model 的支持，不再只是 2026-Q2 的事後猜測。

### 4.4 Full-universe IC 與 Top-10 不對齊

在 117 個真實 validation 交易日：

- Full-universe mean rank-IC：0.0728
- Mean top-10 excess h5 return：+0.300%
- Top-10 與實際報酬前十名的平均重疊率：0.855%
- 約 1,041 檔股票下的隨機重疊期望：約 0.96%

模型能改善全體排序，卻幾乎無法辨識真正的極端贏家。現行 LambdaRankIC
對所有 pair 最佳化，而實際策略只交易前 1%，這是比 pooling 更直接的目標錯配。

---

## 5. 建議執行方案

### Round 6.1：交易日修正與 Feature Ablation（CPU）

不重抽 embedding，直接使用既有 Parquet：

1. 依 `^TWII` calendar 過濾 train/validation rows
2. 每個 fold 邊界 purge 5 個交易日，避免 h5 label 重疊
3. 訓練三個版本：
   - `E`：832 維 embedding only
   - `R`：4 個 raw features only
   - `E+R`：現有完整模型
4. Walk-forward validation：2019、2020、2021、2022、2023、2024H1；
   每個 fold 只使用更早資料訓練

判讀方式：

- `E >= E+R`：raw reversal features 有害，優先移除或降權
- `R ~= E+R`：Kronos embedding 增益有限，M1 架構價值不足
- `E+R` 穩定勝出：保留架構並進入 Round 6.2

### Round 6.2：Top-10 Objective 對齊（CPU）

比較兩種 objective：

1. Baseline：現有 LambdaRankIC
2. Candidate：XGBoost `rank:ndcg`
   - top 1% relevance = 4
   - 1-5% = 3
   - 5-20% = 2
   - 20-50% = 1
   - 其餘 = 0
   - `eval_metric=ndcg@10`
   - pair sampling 使用 `topk`

Rank-IC 保留為 guardrail，但不再是唯一選模指標。主要指標依序為：

1. Median fold top-10 excess return
2. Worst-fold top-10 excess return
3. NDCG@10
4. Mean rank-IC

### Round 6.3：Regime-aware Features（CPU）

從既有 DB 計算後 join 到 Parquet，不需重新跑 Kronos：

1. 橫截面排名：`mom_5/10/20/60_rank`、`ma5/ma20_dist_rank`
2. 風險特徵：`volatility_20_rank`、`volume_ratio_rank`
3. 市場狀態：`TWII momentum_20/60`、`TWII volatility_20`
4. Breadth：全市場高於 MA20 的比例
5. Dispersion：當日個股報酬的截面標準差

每次只加入一組，依序測試：

1. Cross-sectional ranks
2. Multi-horizon momentum
3. Market regime context

不要一次加入全部特徵，否則無法歸因改善來源。

### Round 6.4：Embedding Pooling（GPU，最後才做）

只有 Round 6.1-6.3 證明 embedding 仍有獨立增益後，才重抽：

1. `mean`
2. `last-token`
3. `concat(mean, last-token)`

此階段成本最高，且目前證據顯示 raw features 與 objective mismatch
比 pooling 更值得優先處理。

---

## 6. 統一驗收門檻

候選版本必須：

- 在至少 4 / 6 walk-forward folds 勝過同 fold baseline
- Median top-10 excess 至少改善 +5 bps / h5
- Worst-fold top-10 excess 退步不得超過 10 bps
- Mean rank-IC 不得下降超過 0.01
- 通過 10 / 25 / 50 bps 交易成本壓力測試

候選模型與參數只能依 walk-forward validation 選擇。2024H2 至 2026-Q2
已被多輪研究觀察，只能作最終診斷，不能再用來調參。

## 7. 明確不優先事項

1. 調整 boost rounds：190 與 200 trees 幾乎無差
2. 單獨調整 LambdaRankIC `sigma`：未處理 top-10 目標錯配
3. 立即重抽 last-token / concat embedding：GPU 成本高，且尚未完成 ablation
4. 直接加入大量技術指標：容易把 2026-Q2 的事後解釋寫進模型
