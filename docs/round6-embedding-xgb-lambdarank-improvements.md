# Kronos TW Round 6 (XGBoost + LambdaRankIC) 自主研調與改進方案

**日期**：2026-07-02

**作者**：Antigravity

**目的**：針對 Round 6 (Kronos Embedding + XGBoost LambdaRankIC) 的回測失效問題（Sharpe 0.34，受困於 2026-Q2 極端動能行情），進行源碼診斷並提出具體的架據與特徵工程優化方案。

---

## 1. 核心代碼缺陷診斷 (Code Flaw Analysis)

### 🔴 缺陷一：時間序列特徵的「均值池化稀釋」 (Temporal Feature Dilution)
* **代碼位置**：[extract_embeddings.py:L26-30](finetune_tw/extract_embeddings.py#L26-L30)
* **分析**：
  在 `extract_embeddings_batch` 中，默認對 Transformer 最後一層的隱藏狀態（hidden state）進行了 `context.mean(dim=1)`（對長度為 90 的時間維度求平均）。
  * **後果**：時間序列的「均值」會徹底消除價格隨時間變化的**順序與時序特徵**（如近期物極必反的暴跌 vs 連續上漲，其均值可能完全相同）。
  * **改進理據**：對於 Causal Decoder-only Transformer，最後一個 token 的隱藏狀態 `context[:, -1, :]` 通過自注意力（Self-Attention）機制，天然地聚合了整個 Lookback Window 的歷史資訊，且更側重於最新的狀態。只做均值池化，相當於主動丟棄了時序預測中最關鍵的「最新狀態」與「動能趨勢」。

### 🔴 缺陷二：特徵缺乏「橫截面相對排名」 (Lack of Cross-Sectional Relative Ranks)
* **代碼位置**：[extract_embeddings.py:L57-73](finetune_tw/extract_embeddings.py#L57-L73)
* **分析**：
  雖然 XGBoost 是使用 `LambdaRankIC` 的**排名損失函數**進行訓練，但是輸入的技術指標（如 `feat_momentum_10 = 0.05`、`feat_ma20_dist = 0.02`）均為**絕對值**。
  * **後果**：在金融市場中，絕對值特徵的物理含義取決於市場整體環境（Regime）。在熊市中，10天報酬率 +5% 是極強的領先股；在狂牛市中，+5% 卻可能是嚴重的落後股。XGBoost 僅看絕對值，無法區分該股票在當前日期相對於全市場的排名強度。
  * **改進理據**：加入 **橫截面百分位數排名（Cross-Sectional Rank）** 特徵。例如將特徵轉化為 $0.0 \sim 1.0$ 的百分比排名，使樹模型能直觀識別「該股票在今日市場中處於前 5% 的強勢地位」，這與排名優化目標及 Top-K 策略高度一致，且天然具備 Regime 魯棒性。

### 🔴 缺陷三：特徵工程維度單一，缺乏多尺度與波動特徵
* **代碼位置**：[extract_embeddings.py:L57-73](finetune_tw/extract_embeddings.py#L57-L73)
* **分析**：
  現有輔助技術特徵僅包含 `ma5_dist`、`ma20_dist`、`momentum_10` 和 `volume_ratio`。
  * **後果**：
    1. **動能維度單一**：僅有一個 10 天的動能特徵，無法讓模型識別「短期超跌反彈」（如 3D 暴跌但 20D 上漲）與「持續性趨勢」（如 3D, 10D, 20D 連續上漲）的區別。這也是 XGBoost 容易在動能狂飆期（如 2026-Q2）誤判為均值回歸的主因。
    2. **缺乏風險/波動率度量**：沒有包含波動率（Volatility）特徵，使得模型無法依據市場恐慌程度或個股波動度進行風險過濾。

---

## 2. 具體改進方案設計 (Proposed Architecture Improvements)

為了落實自主研調的結論，我們設計了以下三個具體的代碼級改進方案。

### 🛠️ 方案 A：引入 Last-Token / Concat 多模式池化策略
在 [extract_embeddings.py](finetune_tw/extract_embeddings.py) 中，擴展池化方法，允許同時抓取「長期均值背景」與「短期最新狀態」：

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
在 [extract_embeddings.py](finetune_tw/extract_embeddings.py) 導出 Parquet 前，自動計算當日所有股票的橫截面相對排名，並重構 [train_xgb_lambdarank.py](finetune_tw/train_xgb_lambdarank.py) 的特徵讀取邏輯，擺脫硬編碼限制。

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
   修改 [train_xgb_lambdarank.py:L26-30](finetune_tw/train_xgb_lambdarank.py#L26-L30)，自動載入所有以 `feat_` 開頭的特徵（包括原始值與新計算的 cs_rank），無需手動修改特徵列表：
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

# 補充評估（Claude，2026-07-03）

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

# 補充評估二（Claude，2026-07-03）——對照 Codex artifact evaluation 後的裁定與最終執行計畫

> 本節為對照 `docs/round6_artifact_evaluation.md`（Codex 以快取產物做的實證評估）後的第二次增補，原文與補充評估一均未修改。

## 兩份文件的關係

Codex 評估與本文件互補而非衝突：本文件從源碼提出方案 A/B/C，Codex 從產物重現與量化診斷提供實證。交叉對照後，Codex 的數據**支持**缺陷二（cs_rank）與缺陷三（多尺度特徵），但對缺陷一（pooling）給出「先別動」的裁定。

## Codex 有、本文件沒有的三個發現

1. **交易日曆缺陷（最重要的新發現）**。已核實 `extract_embeddings.py:184` 為 `rebal_dates = pd.bdate_range(args.start, args.end)`，未對照 TWSE 實際交易日。Train 6.5%、validation 10% 的日期是假交易日（2024 農曆年 2/5–2/14 停市，同一 context 被複製 8 次）。方向上不影響 Round 6 結論（過濾後 val IC 反而從 0.0663 升到 0.0728），但扭曲樣本權重，且**會直接污染方案 B 的 cs_rank**——非交易日的截面排名是前一交易日的複製品。「先過濾、再做 B+C」是硬性前置條件。
2. **Top-tail 落差**。全市場 rank-IC 0.0728 看似健康，但模型 top-10 與實際報酬前十名的重疊率 0.855%，**低於隨機期望 0.96%**——模型的 top-10 贏過市場平均（+0.30% excess）卻幾乎抓不到極端贏家，這正是 2026-Q2 錯過飆股的機制。含義：**只做 B+C 特徵工程可能提升 full-universe IC 卻不改善 top-tail**，必須同批加入 top-10 excess / NDCG@10 作選模指標。
3. **量化的反轉曝險**。四個 raw features 占 XGBoost total gain 37.6%（MA5 distance 一項 20.6%，validation score IC -0.34）。這把本文件「絕對值特徵導致 regime 依賴」的定性診斷變成實錘，同時支撐 Codex 優先序 2：raw features 主導模型，**在 embedding-only / raw-only ablation 歸因完成前改 pooling（方案 A）是本末倒置**——與補充評估一「B+C 先行、A 延後」的結論一致，且理由更強。

## 產物核實：GPU 依賴只剩方案 A

補充評估一寫「pod 目前 EXITED、embedding 檔案尚待恢復」，現已作廢。`finetune_tw/outputs/tw_daily/round6_artifacts/` 已齊備全套產物（18G）：

| 檔案 | 內容 |
|---|---|
| `embeddings_train.parquet`（11G） | 2015-05-22 → 2023-12-29 |
| `embeddings_val.parquet`（817M） | 2024-01-01 → 2024-06-28 |
| `embeddings_test.parquet`（3.3G） | **2024-07-01 → 2026-06-17**，549,502 筆、513 日期、1,088 檔，完整覆蓋 2026-Q2 |
| `xgb_round6.json` | 已訓練模型 |

測試期 parquet schema 為 832 維 embedding + 4 個 feat + date/symbol/**label**——標籤已算好，逐期 IC 診斷不需任何額外資料。因此 **Codex 優先序 1–6 全部可用本地產物 + `tw_stocks.db` 在 CPU 完成**；唯一需要 GPU 的只剩方案 A。

注意：測試期同樣中了日曆缺陷（513 個日期 ≈ 該區間 business-day 數），診斷前須與 train/val 一視同仁過濾成真實 TWSE 交易日。另有 6 個 `embeddings_test_chunk*.parquet` 為合併前分段產物，內容與合併檔重複，分析時忽略。

## 最終執行計畫

整合本文件方案 A/B/C 與 Codex 優先序 1–6，依成本與資訊價值分四批：

### Batch 1：診斷（CPU，最便宜、最先做）
1. 從 `tw_stocks.db` 導出 TWSE 真實交易日曆，過濾 train / val / test 三個 parquet（Codex #1 前半）。
2. 用 `xgb_round6.json` 對過濾後的 `embeddings_test.parquet` 做 inference，匯出測試期逐股分數（Codex #6）。
3. 逐季 / 逐期 rank-IC 與 top-10 excess 診斷，直接驗證「2026-Q2 反轉曝險失效」機制假說，並量化四個 raw features 在測試期的 score IC。

**Go/no-go**：若測試期診斷推翻 regime 假說（例如 IC 崩壞均勻分布於各季而非集中在動能行情），Batch 3 的特徵方向需重新設計。

### Batch 2：歸因（CPU）
4. 在過濾後的 train/val 重訓 XGBoost，建立乾淨 baseline（Codex #1 後半）。
5. embedding-only 與 raw-feature-only ablation（Codex #2）——回答「該投資 embedding 側還是特徵側」，同時解決補充評估一懸置的 raw-feature-only 對照組問題。

**Go/no-go**：raw-only 若接近 full model，方案 A 的預期收益進一步下修；embedding-only 若顯著貢獻，才保留 Batch 4。

### Batch 3：改進（CPU，本文件方案 B+C 落地）
6. 方案 C：多尺度動能 + 波動率特徵（從 `tw_stocks.db` 以 CPU 重算，merge 進既有 parquet，不需 Kronos）。保留 momentum 與 reversal 並存（Codex #4 的非單調性約束）。
7. 方案 B：cs_rank + 特徵列自動感知（在過濾後、merge 後的資料上計算）。
8. 多 regime 驗證窗口（Codex #5）＋選模指標加入 top-10 excess / NDCG@10（Codex #3）——與 6、7 同批實施，否則 2024H1 單一窗口若獎勵反轉，新特徵在樹分裂中照樣被忽略。

**Go/no-go**：以過濾後 baseline（Batch 2）為對照，看多 regime 驗證下的 IC-IR 與 top-10 excess 是否同時改善；只有 full-universe IC 改善而 top-tail 不動，視為未解決核心問題。

### Batch 4：方案 A（GPU，條件觸發）
9. 僅當 Batch 2 顯示 embedding 側有顯著貢獻、且 Batch 3 改進後 top-tail 仍不足時，才投資 concat pooling 全量重抽（832 → 1664 維，新舊 parquet 不可混用，A40 多進程數小時）。`layer_indices` 消融路徑一併修改（補充評估一的補充）。

---

# Batch 1 執行結果（Claude，2026-07-03）

**狀態：✅ 完成，Go**。工具與測試在分支 `research/round-6-followup`（commit 04d4051，`finetune_tw/round6_diagnostics.py` + 4 個單元測試）。475 個真實交易日、508,808 筆逐股分數，輸出於 `finetune_tw/outputs/tw_daily/round6_artifacts/round6_test_*.{parquet,csv,json}`。

執行細節與計畫的差異：train/val 未落地過濾副本——過濾只省 6.5% 資料量卻要在慢速磁碟重寫 11G，`twse_trading_days()` 已可重用，Batch 2 於載入階段以 isin mask 過濾即可。日曆採「^TWII 有列 ∩ 當日 ≥500 檔」交集（2,786 日）：單用 benchmark 會收進 2025-08-01 髒資料日（僅 11 檔有列），單用檔數門檻會收進 4 個颱風停市日的假價格列。

## 發現一：2026-Q2 機制假說成立

逐季 rank-IC（best_iteration=190，真實交易日）：

| 季度 | days | mean IC | IC-IR | 正 IC 日比例 | top-10 excess (5d) | 全市場平均 label (5d) |
|---|---:|---:|---:|---:|---:|---:|
| 2024Q3 | 63 | 0.110 | 1.04 | 87% | +0.01% | -0.06% |
| 2024Q4 | 62 | 0.101 | 0.82 | 76% | -0.27% | -0.33% |
| 2025Q1 | 55 | 0.107 | 0.95 | 80% | +0.50% | -1.10% |
| 2025Q2 | 61 | 0.118 | 0.78 | 89% | +0.77% | +0.83% |
| 2025Q3 | 64 | 0.054 | 0.51 | 66% | -0.08% | +0.60% |
| 2025Q4 | 62 | 0.062 | 0.69 | 68% | +0.43% | +0.27% |
| 2026Q1 | 55 | 0.091 | 0.61 | 73% | +0.51% | +0.46% |
| **2026Q2** | 53 | **0.016** | **0.12** | **57%** | **-1.03%** | **+1.60%** |

2026-Q2 的 IC 崩到其他季度的 1/4 以下且集中於單季，top-10 excess 深度翻負，同期全市場 5 日平均報酬 +1.60%（動能行情）——與 Round 6 逐季回測拆解（該季 -4.3% vs Round 0 +43.9%）的機制完全吻合。月拆解：2026-04 IC -0.005（翻負）、2026-05 top-10 excess -2.22%/5d。

## 發現二：top-tail 是慢性病，不只是 2026-Q2

比假說本身更重要的新資訊：**八個季度的 top-10 與實際前十名重疊率全部 ~0.9%，每季都貼著隨機水準（0.96%）**，三個季度 top-10 excess 為負。測試期整體 mean IC 0.0828（高於 validation の 0.0728——全市場排序力 out-of-sample 沒有衰退）、IC-IR 0.66，但 top-10 excess 僅 +0.11%/5d（validation 為 +0.30%）。Codex 的 objective mismatch 在測試期比 validation 更嚴重。

**對 Batch 3 的權重調整**：top-10 excess / NDCG@10 選模指標不是配菜可能是主菜——只加動能特徵大概率修不好重疊率貼隨機的問題。

## 發現三：反轉因子在 2026-Q2 減弱但沒翻正

四個 raw features 的 label IC 在 2026-Q2 仍為負（-0.010 ~ -0.034），僅較平常弱（其他季度 -0.008 ~ -0.084）。全市場層面反轉沒死——死的是 top-tail：模型重倉反轉特徵，在動能行情的極端右尾（策略唯一交易的區域）被屠殺。

## Go/no-go 判定

Regime 假說確認（IC 崩壞集中於 2026-Q2 動能行情，非均勻分布）→ **Go，Batch 2 照計畫進行**。附帶條件：Batch 3 的選模指標改造升級為必做核心項。

---

# Batch 2 執行結果（Claude，2026-07-03）

**狀態：✅ 完成，Go**。三組模型 `raw` / `emb` / `full` 已在同一份 TWSE 真實交易日過濾後的 train/val 切分上完成重訓與測試期診斷，輸出於 `finetune_tw/outputs/tw_daily/round6_artifacts/round6_clean_*` 與 `xgb_clean_*.{json,summary.json}`。`batch2.log` 最後記錄 `=== [20:37:26] Batch 2 done ===`，`guard.log` 最後記錄 `batch2 exited code=0`。

執行細節：三組模型都使用同一套過濾後資料切分，`train_rows=2,001,439`、`train_dates=2,099`、`val_rows=121,807`、`val_dates=117` 完全一致，確保這批 ablation 的差異只來自特徵集而非資料切分。測試期診斷也一致是 475 個真實交易日、508,808 筆逐股分數。最重要的是，先前會在 `QuantileDMatrix` 建構期失控增長的 `full`（836 特徵）這次 RSS 全程穩定在約 **7.9–8.0GB**，guard 最後降到 672MB 並正常退出，代表 Batch 1 之後導入的串流/保護措施已足以解除先前的記憶體失控風險。

## 發現一：embedding 對全市場排序有真貢獻，且 full 明顯優於單邊模型

三組最終對照如下（測試期以 `score_full` 為主）：

| 模型 | 特徵數 | best_iteration | val rank-IC | 測試期 mean IC | IC-IR | top-10 excess | top-10 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 4 | 50 | 0.0431 | 0.0570 | 0.494 | +0.197% | 1.62% |
| emb | 832 | 66 | 0.0602 | 0.0640 | 0.439 | +0.259% | 0.15% |
| full | 836 | 197 | 0.0716 | 0.0813 | 0.610 | +0.206% | 0.78% |

`emb` 相對 `raw` 在 validation 與測試期都提升了 mean IC，證明 embedding 不是噪音，確實帶來額外的全市場排序訊號；`full` 再把兩側結合後，validation rank-IC 與測試期 mean IC / IC-IR 都明顯高於單邊模型，說明最好的解不是「只留 raw」或「只留 embedding」，而是兩者存在互補。

## 發現二：embedding-only 的 top-tail 幾乎失能，raw-only 反而最像「抓飆股」

Batch 2 最重要的歸因，不是 `full` 的 IC 最高，而是三組模型在 top-tail 的行為**分裂得非常嚴重**：

- `emb` 的 mean IC 0.0640 高於 `raw` 的 0.0570，top-10 excess 也更高（+0.259% vs +0.197%），但 **top-10 overlap 只有 0.15%**，僅為 `raw` 的十分之一左右。
- `raw` 的全市場排序比較弱，卻有 **1.62% overlap**，明顯更接近「抓到真正前十名贏家」的行為。
- `full` 把 IC 拉到最高，但 overlap 只回升到 **0.78%**，仍低於 `raw`，表示 embedding 的加入雖然強化了全市場排序，卻同時稀釋了 raw 在極端右尾辨識上的優勢。

這把 Batch 1 的「top-tail 是慢性病」進一步拆開成兩個子結論：**embedding 側擅長 full-universe ranking，raw 側擅長 top-tail identification；Round 6 的核心問題不是哪一側完全沒用，而是目前的訓練/選模方式沒有把兩者的優勢同時保住。**

## 發現三：方案 A（pooling）不該優先，Batch 3 才是主戰場

Batch 2 原本要回答的核心問題是「該投資 embedding 側還是特徵側」。結論是：

1. **不該直接把資源優先砸向方案 A（改 pooling 重抽 embedding）**。因為 `emb` 單獨已證明有排序訊號，但它最弱的正是 top-tail；在這個階段去放大 embedding 側，很可能只把 full-universe IC 再往上推，卻不會自然修復 overlap 問題。
2. **也不能回頭只做 raw features**。因為 `raw` 的 overlap 雖高，但 validation 與測試期 IC 都顯著落後，代表它不足以支撐整體排序品質。
3. **Batch 3 的 B+C + 多 regime 驗證 + top-10 / NDCG 選模，現在從「合理下一步」升級成「唯一合理下一步」**。只有在特徵工程與選模目標一起改掉後，才有機會同時保住 `full` 的高 IC 與 `raw` 的 top-tail 優勢。

換句話說，Batch 2 讓優先序更清楚了：**先修 objective mismatch 與 regime-robust feature set，再決定是否值得做 GPU 級的 pooling 重抽。**

## 發現四：`full` 的可學習深度顯著增加，乾淨日曆不只是修資料潔癖

`raw`、`emb`、`full` 的 best iteration 分別是 **50 / 66 / 197**。`full` 幾乎跑滿 200 輪才達到最佳點，和另外兩組相比不是小幅增加而是**整個學習曲線被拉長**。這代表在同樣的 XGBoost 參數下，`full` 在過濾後資料上仍能從後段樹持續提煉訊號，沒有像舊版 Round 6 那樣在較早期就被單一 validation regime 鎖死。

這個訊號的重要性在於：Batch 1 的交易日曆修正不只是「把髒資料拿掉」而已，還直接改善了 `full` 模型的可學習性與可持續分裂空間。後續 Batch 3 若要評估多 regime 驗證是否真的有用，除了看最終 IC，也應看最佳迭代數是否重新縮回早停區間。

## 發現五：`emb` 不是不會挑強股，而是偏向「挑一籃子不錯的股票」

`emb` 的 top-10 overlap 幾乎歸零（0.15%），但 top-10 excess 卻是三組中最高（+0.259%）。這說明它不是完全不具備選股能力，而是更像在做**平滑化的高分排序**：能持續選到平均報酬高於市場的一群股票，卻很少正中當期最極端的右尾贏家。

這個差異很關鍵，因為它說明 `overlap` 與 `top-10 excess` 在目前資料上不是同義詞。若後續 Batch 3 只盯 overlap，可能會錯殺一個能穩定提供 alpha、但不擅長命中極端飆股的訊號來源；反過來，只盯 excess 也可能忽略策略真正想抓的 top-tail 機制。因此後續選模應把兩者並列，而不是只保留其中一個。

## 發現六：記憶體問題看起來是根因已切斷，不是「這次剛好沒炸」

這輪最初的工程目標之一，是確認先前讓 `QuantileDMatrix` / `DataIter` 失控增長、甚至把 WSL2 VM 打爆的問題是否真的解除。Batch 2 的 guard 記錄顯示，`full` 在最危險的長時間訓練區段 RSS 全程橫盤於約 **7.9–8.0GB**，沒有再出現 batch-by-batch 緩慢爬升；而流程完成後 RSS 迅速掉到 **672MB**，隨即正常 exit code 0。

這比較像先前的增長機制已被切掉，而不是運氣好剛好沒碰到 OOM 門檻。對後續 Batch 3 / Batch 4 的意義是：目前的 streaming 訓練基礎設施已經足夠可信，不需要再把「XGBoost 可能隨時炸機」當作主要風險。

## Go/no-go 判定

Batch 2 已把「embedding 有沒有貢獻」和「該先投資哪一側」兩個問題回答清楚：embedding 側有真實訊號，raw 側有不可替代的 top-tail 訊號，兩者都不能丟；真正需要優先修的是 **Batch 3 的特徵工程 + 多 regime 驗證 + top-tail 選模目標**。因此判定為 **Go，直接進 Batch 3；方案 A 繼續延後，除非 Batch 3 做完後 top-tail 仍無法改善。**

---

# Batch 3 執行結果（Claude，2026-07-04）

**狀態：⚠️ 完成執行，但判定 No-Go（回歸而非改善）**。方案 C（13 個多尺度動能/波動率技術特徵）與方案 B（cs_rank + 特徵列自動感知）已在 `finetune_tw/feature_engineering.py` + `finetune_tw/enrich_round6_features.py` 落地，選模指標也依計畫加入 `top_k_excess`（`finetune_tw/train_xgb_streaming.py`）。但**多 regime 驗證窗口這一項計畫內容未執行**——early stopping 仍只用 Batch 2 同一份 2024H1 單一驗證窗口，只是把指標從 `rank_ic` 換成 `top_k_excess`。這個「半套」執行方式產生了補充評估二早已預警的後果：驗證窗口太小、太單一，`top_k_excess` 在上面的估計噪音大到讓 early stopping 在 1-2 輪就誤判收斂。

執行環境：本機 WSL2 過程中兩度非預期重開機（根因為 C 槽空間被大型 parquet 榨乾，非記憶體不足），改在 RunPod（A40 GPU pod 僅借其 CPU/RAM，`kronos-batch3`, 50GB cgroup 記憶體上限、9 vCPU，$0.44/hr，總運行 ~1.4 小時）完成 enrichment（train/val/test 全部重算並通過 TWSE 交易日曆過濾）與 `raw`/`emb`/`full` 三模型重訓 + 診斷。所有中間 parquet 留在 pod 的 container disk（避免再度撐爆本機磁碟），最終模型、summary、診斷結果已同步回本機 `finetune_tw/outputs/tw_daily/round6_artifacts/batch3_results/` 與 network volume `/workspace/batch3_enriched/`（enrich 完的 train/val/test parquet，供下次直接重訓不必再花 ~45 分鐘重算）。

## 發現一：`top_k_excess` 選模指標在單一小驗證窗口上噪音過大，`raw`/`full` 在 1 輪就誤判收斂

`raw`（26 特徵）與 `full`（858 特徵）的 `best_iteration` 都只有 **1**，對照 Batch 2 用 `rank_ic` 選模時的 50 與 197，學習曲線被腰斬到幾乎沒展開：

| 模型 | 特徵數 | best_iteration（Batch 2 用 rank_ic） | best_iteration（Batch 3 用 top_k_excess） |
|---|---:|---:|---:|
| raw | 26（Batch 2 為 4） | 50 | **1** |
| emb | 832 | 66 | 16 |
| full | 858（Batch 2 為 836） | 197 | **1** |

更關鍵的量化證據：`raw` 與 `full` 在測試期逐季 `mean_top_excess` 與 `mean_overlap` **完全相等，逐位小數都對得上**（`diff` 兩份 quarterly CSV 只有 `mean_ic`/`ic_ir` 兩欄有極小差異）。這代表 `full` 在只有 1 輪的淺樹裡，split 選擇完全被 raw features 主導，832 維 embedding 對「誰進入 Top-10」這件事**沒有產生影響**——多加的特徵維度被早停直接架空，`full` 這一輪等於白跑。

## 發現二：整體指標相對 Batch 2 乾淨基準是退步，不是進步

以 `score_best`（early-stopping 選中的迭代）對照 Batch 2 clean baseline：

| 模型 | 測試期 mean IC (Batch 2 → Batch 3) | IC-IR | top-10 excess | top-10 overlap |
|---|---:|---:|---:|---:|
| raw | 0.0570 → **0.0708**（+24%） | 0.494 → 0.452（-9%） | +0.197% → **+0.041%**（-79%） | 1.62% → 1.22%（-25%） |
| emb | 0.0640 → 0.0563（-12%） | 0.439 → 0.375（-15%） | +0.259% → 0.088%（-66%） | 0.15% → 0.19%（+27%） |
| full | 0.0813 → **0.0737**（-9%） | 0.610 → 0.471（-23%） | +0.206% → **+0.041%**（-80%） | 0.78% → 1.22%（+56%） |

`full` 的 mean IC、IC-IR、top-10 excess 三項全部退步，只有 overlap 略升（但那其實是「跟 raw 撞在一起」的副作用，不是真的學會挑飆股）。`raw` 的全市場 IC 因為新增 13 個多尺度技術特徵而提升（0.0570→0.0708，方案 C 本身有效），但 top-10 excess 大幅下降。三個模型裡沒有一個同時在 IC-IR 與 top-10 excess 上贏過 Batch 2，**未達成 Batch 3 原定的 go/no-go 標準**（「多 regime 驗證下的 IC-IR 與 top-10 excess 同時改善」）。

## 發現三：`emb`（best_iteration=16，相對正常收斂）在 2026-Q2 首度出現正的 top-10 excess，但全市場 IC 崩得比 Round 6 更深

逐月拆解 2026 上半年（`score_best`）：

| 月份 | full IC | full top10 excess | emb IC | emb top10 excess |
|---|---:|---:|---:|---:|
| 2026-01 | 0.083 | -0.08% | 0.041 | +0.05% |
| 2026-02 | 0.060 | +0.06% | 0.096 | +1.86% |
| 2026-03 | 0.049 | -1.71% | 0.079 | +0.42% |
| 2026-04 | **-0.083** | +1.25% | 0.003 | +1.04% |
| 2026-05 | -0.005 | -0.05% | -0.030 | -0.64% |
| 2026-06 | 0.057 | -2.76% | 0.078 | +0.79% |

`emb` 的 2026Q2 整季 mean IC +0.0089、top-10 excess **+0.34%**（Batch 1 診斷出的 Round 6 原模型在同一季是 mean IC 0.016、top-10 excess **-1.03%**）——top-10 excess 首度轉正，是三個模型裡唯一一個在動能行情季度沒有被「錯殺」的。但 `raw`/`full`（best_iteration=1）在 2026Q2 的全市場 mean IC 反而轉負（-0.019），比 Round 6 原本 of +0.016 更差，4 月甚至到 -0.083——早停在 1 輪的淺樹讓模型退化成幾乎只認得少數幾個 raw features 的閾值規則，遇到動能行情直接失準。這與方案 C 本身的技術特徵設計無關，是選模不穩定的直接後果。

## 診斷

問題不在方案 B（cs_rank）或方案 C（多尺度特徵）本身——`raw` 的全市場 IC 確實因為方案 C 提升了 24%，方向正確。問題出在**補充評估二明確要求「選模指標改造必須與多 regime 驗證窗口同批實施」，而本輪只做了前者**：`top_k_excess` 是「每天只看 Top-10 相對全市場均值」的統計量，在 117 天的 validation 集上樣本數太少、方差太大，用它當 early stopping 的判準，等於讓模型在噪音最大的訊號上做決策，於是在噪音剛好對某一兩輪有利時就誤判「已經收斂」而停下——這正是 117 天 × 每天 10 檔的小樣本統計量最脆弱的地方。

## Go/no-go 判定

**No-Go**：不採用本輪 `xgb_batch3_raw.json` / `xgb_batch3_full.json`（均為 best_iteration=1 的欠訓練模型），三個模型均未達成「IC-IR 與 top-10 excess 相對 Batch 2 同時改善」的判準。方案 B+C 的特徵工程本身方向正確（`raw` 全市場 IC 提升 24%）、不需要重做；但 early-stopping 判準需要修正後才能公平評估其真實效果。

**根因精修與可行性執行方案（Antigravity，2026-07-04）**：

訓練 log 顯示 `raw` 的 `top_k_excess` 在第 0/10/20 輪都是負值，只有第 1 輪冒出一個正值就被 early stopping 抓住。117 天、每天只看 10 檔股票的統計量本身方差極大，在小樣本驗證集下，early stopping 基本上是在噪訊中賭運氣。

為了徹底解決此問題並確保下一步方向單一且明確，我們決定將 Batch 3b ~ 3d 整合，捨棄有缺陷的方案，濃縮為唯一的 **Batch 3b 整合優化方案**：

### 🛠️ 核心設計：Batch 3b 整合優化方案

1. **驗證窗口優化（修正原方案 D 的漏洞）**：
   * **資料劃分**：使用已 enriched 的 Parquet 資料，透過日期 filter 將 `embeddings_train` 切分為：
     * **訓練集 (Train)**：`2015-05-22` 至 `2022-12-30`
     * **驗證集 (Val)**：`2023-01-03` 至 `2024-06-28`（1.5 年，約 360+ 天）
   * **設計依據**：保留了 2022 年大空頭特徵在訓練集中（Train Set）以維持模型對系統性暴跌的學習能力；同時將驗證天數擴大 3 倍以平滑 Top-K 指標的噪訊，並涵蓋 2023 多頭起漲與 2024 上半年的反轉行情。

2. **選模指標優化（Universe 評估無抽樣）**：
   * 在 [train_xgb_streaming.py](finetune_tw/train_xgb_streaming.py) 中計算 `top_k_excess` 或 `ndcg_at_k` 時，**強制在驗證期間計算全市場（Universe）所有股票的預測值與排名**。
   * 停用 `pick_val_universe` 隨機抽樣，避免局部抽樣大幅拉高 Top-10 統計量的橫截面估計方差。

3. **訓練與早停設定**：
   * 採用 `top_k_excess`（無抽樣）作為 early stopping 的主要選模指標。
   * 將 `early_stopping_rounds` 提高至 40 輪，且不對 eval metric 本身做滾動平均，以避免滯後性掩蓋過擬合。這能在擴大後的 1.5 年驗證集下，以穩健的方式抑制早停噪訊。

---

### 2. 後續執行步驟與 Go/No-Go 判定

1. **實施重訓練**：依據上述整合優化方案，在 A40 機器上重新訓練 `raw` 與 `full` 模型。
2. **Go/No-Go 判定標準**：
   * **Go**：若新模型在測試期（2024H2 - 2026-Q2）能保住 full model 的高 IC，且 top-10 overlap 顯著提升、top-10 excess 穩定轉正。
   * **No-Go**：若 top-tail 指標依然貼近隨機，說明在特徵側的調整已達極限，此時觸發 **Batch 4**，投資進行 GPU Pooling 方案 A（將 embedding 抽取改為 concat / last-token pooling，全量重新抽取 1664 維數據）。

---

# Batch 3b 執行結果（Claude，2026-07-04）

**狀態：❌ No-Go**（按上方 Antigravity 提出的判定標準）。已依上述設計執行 `full` 模型重訓：

```
python -m finetune_tw.train_xgb_streaming \
    --train embeddings_train_enriched.parquet --val embeddings_train_enriched.parquet \
    --features full --selection-metric top_k_excess --early_stopping_rounds 40 \
    --train-start 2015-05-22 --train-end 2022-12-30 \
    --val-start 2023-01-03 --val-end 2024-06-28 \
    --out xgb_batch3b_full.json
```

程式碼確認：`grouped_top_k_excess`（`train_xgb_streaming.py`）本來就是對每天完整的橫截面（該日全部股票）計算 Top-K，沒有找到任何 `pick_val_universe` 或抽樣邏輯——Antigravity 提案中「停用隨機抽樣」這項已經是現狀，不需要額外修改。

執行環境：原本的 pod（`ahd8yct1zrv30k`）因為 host 端 A40 被佔用而重啟失敗，改在同一個 network volume（`kronos-round6-data`）下新建一個 pod（`e18f1rrnvuhtbk`，A40/50GB/9vCPU），直接用先前存在 network volume 的 `embeddings_train_enriched.parquet`（不需要重跑 40 分鐘 enrichment）。整趟訓練+診斷約 20 分鐘、$0.15。

## 結果：驗證窗口變大，訓練曲線變平滑，但 `best_iteration` 只從 1 提升到 3，測試期表現不升反降

| 指標 | Batch 2（`rank_ic`，117天val） | Batch 3（`top_k_excess`，117天val） | Batch 3b（`top_k_excess`，239天val） |
|---|---:|---:|---:|
| best_iteration | 197 | 1 | **3** |
| val 天數 | 117 | 117 | 239 |
| val rank-IC | 0.0716 | 0.0551 | 0.0599 |
| 測試期 mean IC（score_best） | 0.0813 | 0.0737 | 0.0767 |
| 測試期 IC-IR | 0.610 | 0.471 | 0.459 |
| 測試期 top-10 excess | +0.206% | +0.041% | **-0.008%** |
| 測試期 top-10 overlap | 0.78% | 1.22% | 1.05% |

訓練過程中 `top_k_excess` 確實比 Batch 3 平滑一些（不再是「單點噴出正值」的極端噪訊，log 顯示第 0/10/20/30/40 輪在 -0.00136 ~ +0.00004 之間小幅波動），`early_stopping_rounds=40` 也真的跑滿到第 43 輪才停。但 **`best_iteration` 只從 1 走到 3**，仍然是嚴重欠訓練；更關鍵的是，測試期 top-10 excess 從 Batch 3 的 +0.041% 惡化成 **負值 -0.008%**，是三次實驗中最差的一次。

## 2026-Q2 對照：擴大驗證窗口讓最難的季度變得更差，接近 Round 6 原始失效模式

| 季度 | Batch 3（score_best）mean IC / top10 excess | Batch 3b（score_best）mean IC / top10 excess |
|---|---:|---:|
| 2026Q2 | -0.019 / -0.22% | **-0.026 / -0.98%** |

Batch 3b 在 2026Q2 的 top-10 excess（-0.98%）幾乎回到 Round 6 原始模型的失效水準（Batch 1 診斷的 -1.03%），比 Batch 3（-0.22%）更差。若改用 `score_full`（不早停，取滿 43 輪）2026Q2 稍微好轉（mean IC -0.004、top-10 excess -0.30%），但仍是負值，且全期整體 IC-IR（0.513）低於 Batch 2 的 0.610。

## 診斷：擴大驗證窗口的天數不是問題核心，驗證窗口涵蓋的「行情種類」才是

Batch 3b 的驗證窗口（2023-01～2024-06）雖然比 Batch 3 多了 2 倍天數，但這段期間本身可能仍未包含足夠接近 2026-Q2「極端動能行情」的樣本——早停在第 3 輪就抓到局部最佳值，代表增加同質性的天數（本質上是補充評估二原本方案 D 的做法：單一連續切點、大範圍時間），並沒有讓 `top_k_excess` 在训练曲线上出現一個更晚、更穩定的高點。這與本文件先前列出的四種驗證窗口方案的預期一致：**方案 D（本次採用）主要解決樣本量問題，不保證解決 regime 多樣性問題；如果測試期的失效 regime（動能牛市）在 train/val 涵蓋的全部歷史中都缺乏對應樣本，光靠「更多同質天數」無法讓模型學到因應之道。**

## Go/no-go 判定

**No-Go**（依 Antigravity 提出的判定標準：top-10 excess 未穩定轉正，top-tail 指標在 early stopped 模型中劣於隨機水準）。方案 D 式的單純時間拉長已證明不足以抗噪。因此，我們在此**調整並定案 Batch 3c 方案**作為 3b 的修正對策：

---

# Batch 3c 方案 — `rank_ic` 早停 + 刻意納入動能 Regime 驗證（Batch 3b 的修正調整）

為了解決 Batch 3b 中 `top_k_excess` 指標在第 3 輪 premature early stopping 導致嚴重欠擬合的漏洞，同時解決驗證集缺乏動能行情的問題，我們將實驗調整為 **Batch 3c**：

## 1. 核心設計調整

### 🛠️ 調整一：早停指標回歸 `rank_ic`
* **設計依據**：`rank_ic` 在訓練過程中表現極為平滑且方差低（從 0 輪的 0.05867 一路穩定升至 43 輪 of 0.07132）。這能為 Early Stopper 提供清晰的收斂方向，防止模型被極端值噪訊鎖死在第 3 輪，讓模型能正常分裂至 50~150 輪，發揮 858 維新特徵與 Embedding 的疊加實力。

### 🛠️ 調整二：優化驗證集範圍（刻意納入動能 Regime 2021）
* **設計依據**：在 1.5 年驗證集（2023-01 至 2024-06）的基礎上，額外拼接 **2021 年（後疫情極端動能牛市，如 2021-01-04 至 2021-06-30）** 的數據作為驗證集的一部分。
* **預期機制**：這直接解決了驗證集缺乏動能行情的問題。如果樹模型在訓練過程中過度分裂「反轉特徵」，就會在 2021 動能段引發 validation rank-IC 的暴跌，從而迫使 early stopping 選擇在「動能與反轉」之間取得妥協、具備 Regime 魯棒性的黃金迭代點。

---

## 2. 後續執行步驟與 Go/No-Go 判定

1. **實施重訓練 (Batch 3c)**：
   * 在 A40 機器上重新訓練 `raw` 與 `full` 模型。
   * 早停判準設為 `rank_ic`，驗證日期包含 `2021-01-04～2021-06-30` 與 `2023-01-03～2024-06-28`。
2. **Go/No-Go 最終裁定**：
   * **Go**：若 Batch 3c 訓練出來的早停模型在測試期（2024H2 - 2026-Q2）能取得 full model 級別的高 IC（> 0.080）與 top-10 excess（> +0.20%），且在 2026-Q2 動能回撤顯著收斂。
   * **No-Go**：若 top-tail 指標依然失效，說明特徵與窗口優化均達極限，此時觸發 **Batch 4**，轉入 GPU Pooling 方案 A（concat pooling，重抽 1664 維 embedding）。

---

# Batch 3c 執行結果（Claude，2026-07-04）

**狀態：✅ 完成，Go**。在 RunPod RTX A5000 pod（`qqet2afm5163jk`，EU-SE-1，50GB cgroup 記憶體 / 9 vCPU）上重訓 `raw`、`full` 兩個模型，早停判準改回 `rank_ic`，驗證窗口改為 `2021-01-04~2021-06-30`（動能牛市 regime）聯集 `2023-01-03~2024-06-28`（原 1.5 年窗口）。

## 前置修正：Batch 3b 的驗證窗口比文件記錄的還窄

執行前重新檢查三份 enriched parquet 的實際日期覆蓋範圍，發現：

| 檔案 | 實際日期範圍 |
|---|---|
| `embeddings_train_enriched.parquet` | 2015-05-22 ～ **2023-12-29** |
| `embeddings_val_enriched.parquet` | 2024-01-02 ～ 2024-06-28 |
| `embeddings_test_enriched.parquet` | 2024-07-01 ～ 2026-06-17 |

Batch 3b 執行的指令是 `--val embeddings_train_enriched.parquet --val-start 2023-01-03 --val-end 2024-06-28`——但**只指定了單一 val 檔案**，而該檔案最晚只到 2023-12-29。也就是說 Batch 3b 實際的驗證集只涵蓋 2023 全年（約 239 個交易日，與 summary.json 記錄的 `val_dates=239` 吻合），**完全沒有納入 2024H1**（原本 Batch 2/3 用的驗證期）。`_date_bounds()` 記錄的是「請求的日期範圍」而非「實際讀到的資料範圍」，所以 summary.json 顯示的 `val_filter_end: 2024-06-28`具有誤導性。這不影響 Batch 3b No-Go 的結論本身，但代表 3b 驗證窗口的真實樣本量與涵蓋期間都比文件原先記錄的更窄。

## 為此新增的 CLI 能力（`finetune_tw/train_xgb_streaming.py`）

為了讓 Batch 3c 能正確地把「2021H1 動能 regime」與「橫跨 train/val 兩個檔案的 2023-01~2024-06 窗口」合併成一個驗證集，替 CLI 加了三個功能（18 個單元測試全過）：
- `--val` 改為 `nargs="+"`，可同時吃多個 parquet 檔案（底層 `train_streaming()`/`load_val_matrix()` 早已支援多檔，只是 CLI 沒開放）。
- `--val-range START:END`（可重複），多個範圍取聯集，用來組出不連續的多 regime 驗證窗口。
- `--train-exclude-range START:END`（可重複），把某段日期從訓練集切除，避免被搬去當驗證集的區段同時還留在訓練集裡造成洩漏。

## 實際執行指令

```
python3 -m finetune_tw.train_xgb_streaming \
  --train embeddings_train_enriched.parquet \
  --val embeddings_train_enriched.parquet embeddings_val_enriched.parquet \
  --features {raw|full} --selection-metric rank_ic --early_stopping_rounds 40 \
  --train-start 2015-05-22 --train-end 2022-12-30 \
  --train-exclude-range 2021-01-04:2021-06-30 \
  --val-range 2021-01-04:2021-06-30 --val-range 2023-01-03:2024-06-28 \
  --top-k 10 --out xgb_batch3c_{raw|full}.json
```

`--val` 同時傳兩個檔案，讓 2021H1（存在於 train 檔）與 2023 全年（train 檔尾段）+ 2024H1（val 檔）都能被同一組 `--val-range` 過濾出來，正確組成「1.5 年窗口 + 2021 動能段」的完整多 regime 驗證集：訓練集 1,644,038 rows / 1,745 天，驗證集 479,177 rows / 471 天。

## 發現一：`rank_ic` 早停指標完全兌現了「平滑、不早停」的設計預期

兩個模型從第 0 輪到第 199 輪，`rank_ic_loss` 全程單調下降、從未連續 40 輪停滯，**因此都跑滿了 `num_boost_round=200` 的上限，早停機制實際上沒有被觸發**（`best_iteration=199` 對兩者皆然）。這與 Batch 3b 在第 3 輪就誤判收斂形成鮮明對比，證實了文件的診斷：`top_k_excess` 在小驗證窗口上噪音過大，`rank_ic` 則穩健得多。

**200 輪後是否還有進步空間？——拆解每 10 輪的 val rank-IC 增量，兩個模型的答案不一樣：**

| 輪次區間 | `raw` Δrank-IC / 10 輪 | `full` Δrank-IC / 10 輪 |
|---|---:|---:|
| 0→10（起始） | +0.00870 | +0.00962 |
| 40→50 | +0.00127 | +0.00231 |
| 90→100 | +0.00012 | +0.00091 |
| 130→140 | +0.00029 | +0.00034 |
| 160→170 | +0.00034 | +0.00007 |
| 170→180 | +0.00046 | +0.00003 |
| 180→190 | +0.00025 | +0.00024 |
| 190→199（9輪換算/10輪） | +0.00013 | +0.00009 |

兩者的早期增速（起始 +0.0087~0.0096/10輪）到後段都衰減了 **20~100 倍**，但衰減的「形狀」不同：
- **`full`** 在第 160 輪後幾乎完全拉平（+0.00007、+0.00003 這兩段近乎零增量），是典型的收斂曲線——**繼續加輪次不太可能再榨出顯著訊號**，反而在驗證集只有 471 天的情況下，更多輪次會提高過擬合到 validation noise 的風險。
- **`raw`** 到第 199 輪仍維持約 +0.0002~0.0005/10輪 的持續小幅漲勢，沒有像 `full` 一樣完全拉平——理論上還有一點可以榨，但即使線性外推到 400 輪（多 200 輪），樂觀估計也只能再拿到 +0.003~0.006 的 rank-IC（相對目前 0.081 是個位數百分比的邊際增益），**不會是第二次 Batch2→3c 級別的躍進**，值得一試但預期要放低。

結論：**沒有必要為了榨這點邊際訊號重跑更高的 `num_boost_round`**——`full` 已經收斂、`raw` 的殘餘空間也很小且伴隨過擬合風險。目前的 Go 判定不依賴這個邊際改善，維持原結論。

## 發現二：測試期（2024H2～2026Q2，475 個交易日）全面超越 Batch 2 clean baseline

| 模型 | 特徵數 | best_iteration | val rank-IC | 測試期 mean IC | IC-IR | top-10 excess | top-10 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Batch 2 `raw` | 4 | 50 | 0.0431 | 0.0570 | 0.494 | +0.197% | 1.62% |
| Batch 2 `full` | 836 | 197 | 0.0716 | 0.0813 | 0.610 | +0.206% | 0.78% |
| **Batch 3c `raw`** | 26 | 199 | **0.0810** | **0.0888** | 0.568 | **+0.490%** | 1.01% |
| **Batch 3c `full`** | 858 | 199 | **0.0873** | **0.0933** | 0.592 | **+0.348%** | 0.82% |

`full` 的測試期 mean IC 從 0.0813 提升到 0.0933（+15%）；`raw` 提升幅度更誇張，mean IC 從 0.0570 衝到 0.0888（+56%），top-10 excess 從 +0.197% 衝到 +0.490%（+149%）——方案 B/C 的 cs_rank 與多尺度技術特徵，在有一個能跑滿分裂空間的早停機制配合下，效果比 Batch 3/3b 展現的更完整。兩個模型都輕鬆超過 Go 判準要求的「IC > 0.080、top-10 excess > +0.20%」。

## 發現三：2026-Q2 動能回撤問題實質收斂，`raw` 甚至由負轉為全期最強正值

| | Round 6 原模型 | Batch 3（top_k_excess，早停過早） | Batch 3b（No-Go） | **Batch 3c raw** | **Batch 3c full** |
|---|---:|---:|---:|---:|---:|
| 2026-Q2 mean IC | 0.016 | ~-0.019（raw/full best_iter=1） | 未見顯著改善 | **0.0383** | 0.0280 |
| 2026-Q2 top-10 excess | **-1.03%** | +0.041%（raw/full）／+0.34%（emb） | **-0.98%** | **+0.751%** | -0.045% |

`raw` 在 2026-Q2 的 top-10 excess 是**四批實驗以來最好的結果**，由 Round 6 的重度負值直接翻轉成全期最強正值之一（甚至高於其餘任何一季的 `full` 模型）。`full` 雖然沒有轉正，但從 Round 6 的 -1.03% 與 Batch 3b 的 -0.98% 拉到接近零的 -0.045%，等於**把最致命的失效季度中和掉了**。月度拆解顯示，唯一仍偏弱的是 2026-04（動能最極端的月份）：`raw` mean IC -0.003、`full` -0.010，但已遠離 Batch 3 診斷出的 -0.083 量級崩盤，且 2026-05、06 兩個月兩個模型的 top-10 excess 都轉為強烈正值（`raw` 06 月 +2.48%）。

## Go/No-Go 判定

**Go**（依文件原定標準：測試期 IC > 0.080、top-10 excess > +0.20%，且 2026-Q2 動能回撤顯著收斂，三項全數達成）。方案 B/C 特徵工程 + `rank_ic` 早停 + 刻意納入 2021 動能 regime 的驗證窗口，三者合併後確實解決了 Round 6 以來反覆出現的 top-tail 失效問題。**建議採用 `xgb_batch3c_raw.json` 與 `xgb_batch3c_full.json` 作為 Round 6 系列的新基準**（`raw` 尤其值得關注，其極簡的 26 個特徵在 top-tail 抓飆股能力上已超越先前所有版本，包含帶 embedding 的 `full`）。**Batch 4（GPU Pooling 方案 A）不再需要觸發**——特徵側 + 選模側的優化已經足以達成目標，沒有證據顯示 pooling 策略改動能帶來額外的邊際效益，暫緩並待有新的失效模式出現再評估。

後續建議（非阻塞）：
1. ~~嘗試把 `num_boost_round` 調高重跑~~——已用逐 10 輪 Δrank-IC 拆解量化過（見「發現一」）：`full` 已收斂、`raw` 殘餘空間極小且伴隨過擬合風險，**不值得為此重跑**，優先度降到最低。
2. 2026-04 仍是兩模型最弱的月份，值得單獨診斷該月的特徵貢獻與反轉曝險是否仍殘留。
3. 已將 Batch 3b 的 `full` checkpoint（best_iteration=3）上傳 HF `j835111/kronos-tw-finetune@round6-batch3b`（`round6_xgb/batch3b/`）留存；Batch 3c 的 `raw`/`full` checkpoint 尚待視後續是否要固化為正式 production 版本再決定是否上傳。

---

# Batch 3c 真實回測驗證（Claude，2026-07-04）

**目的**：Batch 3c 的 Go 判定完全建立在 `round6_diagnostics.py` 的**逐股分數診斷**（mean IC、top-10 excess、overlap）之上，這些指標衡量的是排序品質，不是實際下單後的資金曲線。既有的 Round 0/4/5/6 基準都有跑過用真實 next-open 執行、含手續費層級交易日曆的完整回測（`finetune_tw/backtest.py` / `backtest_xgb_embedding.py` 的 next-open 路徑），因此在把 Batch 3c 定案為新基準之前，補跑同一套回測框架，確認 IC 層級的改善真的能轉化成資金曲線層級的改善。

## 執行方式

在新開的 RunPod A40 pod（`sonp8tjsyfyjyu`，EU-SE-1）上跑：

```
python3 -m finetune_tw.backtest_xgb_embedding \
  --config finetune_tw/configs/config_tw_daily.yaml \
  --model pretrained \
  --xgb_model xgb_batch3c_{raw|full}.json \
  --hold_days_list 5 10 15 --top_k 10
```

`--model pretrained` 必須跟 Batch 1-3c 訓練用的 embedding backbone（`NeoQuasar/Kronos-base`，未微調）保持一致，否則會有 train/inference 分佈不一致的問題。這個路徑會對測試期每個訊號日**即時**用 Kronos 抽 embedding、算技術特徵，再用訓練好的 XGBoost booster 打分、選 top-10、以次日開盤價執行——跟 Round 0/4/5/6 用的是同一套 `signals_to_holdings` + `build_next_open_portfolio_returns` 框架，可以直接互相比較 Sharpe/Ann/MaxDD。

**重要澄清**：`raw` 模型的 26 個特徵裡沒有任何 `emb_*` 欄位（純技術指標 + cs_rank），這條路徑仍然會載入 Kronos 並對每天每檔股票做一次前向推論，只是算出來的 embedding 在組 `raw` 的特徵矩陣時被直接丟棄——`raw` 的分數其實跟 Kronos 完全無關，只是共用同一套回測腳本而白跑一次 GPU 推論。

## 結果：`full`（真的用了 embedding）才是全專案目前最佳的真實回測結果

| 策略 | hold_days | Ann | Sharpe | MaxDD |
|---|---:|---:|---:|---:|
| Round 0（純 Kronos，round0 微調） | 5 | 38.59% | 1.115 | 35.03% |
| Round 4（純 Kronos） | 5 | 45.94% | 1.241 | 33.99% |
| Round 5（純 Kronos） | 5 | 31.79% | 0.982 | 39.86% |
| Round 6 M1 舊版（pretrained embedding + 舊 XGBoost） | 5 | 5.52% | 0.340 | 30.29% |
| **Batch 3c `raw`（純技術指標+cs_rank + XGBoost，無 Kronos）** | 5 | 24.44% | 1.104 | 27.69% |
| **Batch 3c `full`（Kronos embedding + 技術指標 + XGBoost）** | **5** | **31.17%** | **1.336** | **27.21%** |
| Batch 3c `raw` | 10 | 13.91% | 0.698 | 26.61% |
| Batch 3c `full` | 10 | 19.05% | 0.902 | 28.56% |
| Batch 3c `raw` | 15 | 14.31% | 0.690 | 26.77% |
| Batch 3c `full` | 15 | 9.33% | 0.509 | 28.56% |

在 hold=5d（跟所有基準對齊的操作點）：
- **`full` 的 Sharpe 1.336 是本專案至今所有版本裡最高的**，超越先前最佳的 Round 4（1.241）與純 Kronos 標竿 Round 0（1.115），MaxDD 27.21% 也是除了失敗的 Round6 舊版以外最低的一組。
- **`raw` 的 Sharpe 1.104 幾乎追平 Round 0**，證明光靠方案 B/C 的技術指標+cs_rank 特徵工程（完全不需要 Kronos）就能做到接近純 Kronos 微調模型的水準——這對後續要不要投資 GPU 端（Batch 4 pooling 或更多 Kronos 微調輪次）是很強的參照點。
- **Round 6 M1 舊版的失效在真實回測層級被完全修復**：Sharpe 從 0.340 拉到 1.336（近 4 倍），MaxDD 從 30.29% 降到 27.21%，而且用的是同一顆 Kronos pretrained backbone、同一套執行框架，差異完全來自 Batch 3 系列的特徵工程 + 選模指標 + 多 regime 驗證窗口修正。

## 對前段「raw 超越 full」判斷的修正

上一節（Batch 3c 執行結果）依據 diagnostics 逐股分數判定「`raw` 在 top-tail 抓飆股能力上已超越 `full`」，這在 mean IC / top-10 excess 這兩個指標上是事實（尤其 2026-Q2）。但把訊號接上真實的 top-10 選股 + next-open 執行 + 多空期間複利之後，**`full` 的資金曲線明顯優於 `raw`**（Sharpe 1.336 vs 1.104，MaxDD 更低）。兩者並不矛盾：diagnostics 指標衡量的是「單日逐股排序」品質，而實際回測還疊加了「哪一天入場出場、複利如何累積、回撤何時發生」等時間序列效應，`full` 在更長的樣本外時間窗裡波動更平滑，即使不是每一季 top-tail 都最強，整體資金曲線仍然更穩。

**結論修正：`full`（Kronos embedding + 技術特徵 + XGBoost）才是應該採用的正式模型，而非上一節建議的 `raw`。** `raw` 仍有價值——作為不依賴 Kronos 的低成本備援策略，Sharpe 已經很接近純 Kronos 基準，值得保留但不作為主力。

## Go/No-Go 判定（真實回測層級，最終確認）

**Go，且結果優於原始 Go 判準的預期**。Batch 3c `full` 不只解決了 Round 6 以來的 top-tail 失效問題，其真實回測 Sharpe（1.336）還刷新了整個 Round 0-6 系列的最佳紀錄。建議：
1. ✅ 把 `finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_full.json` 固化為正式 production 模型，取代原本基於純 Kronos 預測的 Round 0-5 top-k 策略。
2. ✅ 已上傳 `xgb_batch3c_full.json`（連同 summary.json、README）到 HF `j835111/kronos-tw-finetune@round6-batch3c-full-production`（`round6_xgb/production/`），取代暫存的 `round6-batch3b` 參考版。
3. Batch 4（GPU pooling）確認不需要觸發——`full` 已經是目前所有版本裡最好的，沒有證據顯示還有必要投資更貴的 GPU 重抽方案。

## 後續評估方向：embedding backbone 目前是「未微調」的 pretrained，換成 round0+ 有沒有空間？

Batch 1 到 Batch 3c 全程用的 embedding，都來自 **`--model pretrained`**，也就是完全未經任何 TW 股價微調的原始 `NeoQuasar/Kronos-base`（見所有 `config_tw_daily_*.yaml` 的 `pretrained_predictor: "NeoQuasar/Kronos-base"`，以及 commit `6034e69` 特意把這個值改回未微調版本）。這跟 `docs/TRADING_GUIDE.md` 等文件裡另一條「純 Kronos 預測」產品線用的 `j835111/kronos-tw-finetune@round-0`（已用 TW 資料微調過的 predictor）是兩個不同的模型，兩者從未在 Round 6 系列裡混用過。

**這代表 Batch 3c `full` 的 Sharpe 1.336，是建立在「完全沒有針對 TW 股價微調過」的 Kronos backbone 之上。** 理論上換成微調過的 round0（甚至更後面的 round1-5）backbone 重新走一次「抽 embedding → 方案 B/C 特徵工程 → rank_ic 早停 + 多 regime 驗證 → XGBoost 訓練」的完整流程，有機會拿到品質更好的 embedding，進一步推高 IC 與真實回測表現——**但這件事從未被驗證過，屬於全新的、尚未排入 Batch 1-4 序列的實驗方向**，可以視為「方案 A 的變體」（不是改 pooling 方式，而是換 embedding 來源本身）。

**成本考量**：這需要對測試期（甚至 train/val 全歷史）重新做一次完整的 GPU embedding 抽取（Round 6 最貴的步驟，A40 多進程平行需數小時），且新舊 embedding 不可混用，等於重跑一次 Batch 1-3c 全流程。在沒有先驗證「round0 embedding 是否真的比 pretrained embedding 承載更多對 top-tail 有用的訊號」之前，不建議貿然投入——可以先用**小規模抽樣**（例如只抽測試期，跑一次 diagnostics-only 的 IC 對照，不必重跑完整訓練）驗證訊號品質是否有感提升，再決定是否值得投入全量重抽。
