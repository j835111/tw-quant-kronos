# Kronos-TW 上游 PR 採納設計（下一階段落實）

**日期**：2026-06-19
**範圍**：`finetune_tw/`（台股日線、cross-sectional top-K 策略）
**來源**：原始 repo `shiyu-coder/Kronos` 的社群 PR，挑選對台股調整方向有幫助者，對照本地程式碼確認問題是否實際存在。
**目標**：提升回測可信度（消除資料洩漏）→ 提升訊號品質 → 加速 Colab 迭代。

---

## 0. 結論摘要（先看這個）

對照本地程式碼後，**確認存在的問題**有 3 個是「現在就該修」的硬傷：

| # | 問題 | 證據（本地檔案） | 對應上游 PR | 嚴重度 |
|---|------|----------------|------------|--------|
| P0-1 | 正規化用「整個 window（含預測期）」算 mean/std → **未來資訊洩漏** | `dataset.py:56-58` | #263 / #234 / #83 | 🔴 致命：回測 Sharpe 虛高 |
| P0-2 | `amount` 欄位永遠寫死 0.0 → 6 個特徵有 1 個是死特徵 | `yfinance_fetcher.py:51` | #311（成交額代理） | 🔴 高：浪費 1/6 輸入維度 |
| P0-3 | predictor 訓練無 AMP/混合精度（CLAUDE.md 宣稱有，實際無） | `train_predictor.py` 無 `autocast` | #288 / #289 | 🟡 中：Colab 訓練慢、吃顯存 |

**值得做但非急迫**：回測加速（#53 批次預測 / #192 KV cache）、機率排序選股（#321）、core sampling bug 對照（#238 / #262 / #244）。

---

## 1. P0-1：消除正規化資料洩漏（最高優先）

### 問題
`finetune_tw/dataset.py` `MultiStockDataset.__getitem__`：

```python
# dataset.py:53-58 — 現況（有洩漏）
x = self._data[sym][start : start + self.window].copy()   # window = lookback + predict + 1
...
mean = x.mean(axis=0)        # ← 對「整個 window」算統計量
std  = x.std(axis=0) + 1e-5  # ← 含了預測目標期的價格分佈
x = np.clip((x - mean) / std, -self.clip, self.clip)
```

`window = lookback_window + predict_window + 1`（config 為 90 + 10 + 1 = 101）。
mean/std 把後 11 根（預測目標期）的價格統計也算進去，等於把「未來的均值與波動」洩漏進輸入特徵的正規化參數。模型可間接推知未來價格區間 → **回測指標系統性虛高**。

上游 #263 指出 sibling `QlibDataset` 的正確做法是只用 lookback 段，#234 已將同類修復併入主線。

### 修復
只用 lookback 段算統計量，再套用到整個 window：

```python
# dataset.py — 修正後
x = self._data[sym][start : start + self.window].copy()
s = self._stamps[sym][start : start + self.window].copy()

past = x[: self.lookback_window]          # 只看歷史段
mean = past.mean(axis=0)
std  = past.std(axis=0) + 1e-5
x = np.clip((x - mean) / std, -self.clip, self.clip)
```

需在 `__init__` 存下 `self.lookback_window = lookback_window`（目前只存了 `self.window`）。

### 一致性檢查
- `backtest.py` 走的是 `KronosPredictor.predict()`，正規化由 model 端負責，**不經過此 dataset**。確認 predictor 的 normalize 只用 context 段（KronosPredictor 預設行為），避免「訓練修好、推論又洩漏」的不對稱。
- 修完後**訓練/驗證 loss 的絕對值會變化**，這是預期的；重點看回測指標是否回落到合理區間。

### 驗收
- 新增單元測試：建構一個已知 array，斷言正規化後**預測期那幾根不影響 mean/std**（mean/std 只由前 `lookback_window` 根決定）。
- 重跑一輪訓練 + 回測，記錄修復「前 vs 後」的 Sharpe / Annual Return。預期後者較低但**可信**。

---

## 2. P0-2：成交額 `amount` 用代理值取代寫死的 0（借鏡 #311）

### 問題
```python
# yfinance_fetcher.py:51 — amount 永遠是 0
"amount": 0.0,
```
`FEATURES = ["open","high","low","close","volume","amount"]`（`dataset.py:8`，對應 `d_in=6`）。
`amount` 全為 0 → 第 6 個輸入特徵是常數，等於白白浪費 1/6 的模型輸入維度，且正規化時 std=0（靠 `+1e-5` 勉強不爆）。

上游 #311（印度 pipeline）的做法：`amount = volume × avg(OHLC)` 當成交額代理。台股可直接套用。

### 修復
```python
# yfinance_fetcher.py — amount 用成交額代理
avg_price = (hist["Open"] + hist["High"] + hist["Low"] + hist["Close"]) / 4.0
df = pd.DataFrame({
    ...
    "volume": hist["Volume"].values,
    "amount": (hist["Volume"] * avg_price).values,   # 成交額代理（股數 × 均價）
})
```

### 注意事項
- 其他 fetcher（`finmind_fetcher.py`、`twse_scraper.py`）若有**真實成交金額**，應優先用真值；只有 yfinance 缺欄位時才用代理。需確認三個 fetcher 的 `amount` 語意一致（同為「金額」而非「張數」）。
- 此改動會讓**已下載的 DB 需要重抓或回填** amount 欄位（用 `--update` 或重建）。
- 改完特徵分佈變了，**tokenizer 需重訓**（Cell 5+6），不能只重訓 predictor。

### 驗收
- 抽查 2330.TW 某日 `amount` 量級是否合理（百億級新台幣）。
- 對照含/不含真實 amount 的版本，看 tokenizer 重建誤差與回測指標。

---

## 3. P0-3：predictor 訓練加 bf16 混合精度（借鏡 #288 / #289）

### 問題
`train_predictor.py` 的訓練迴圈是 full fp32，無 `torch.autocast` / `GradScaler`。
CLAUDE.md 描述為「frozen tokenizer, AMP」，但**程式碼實際沒有 AMP**——文件與實作不一致。Colab T4/A100 上 fp32 既慢又吃顯存，拖慢迭代循環。

### 修復
參考 #288，在 config 加 `amp_dtype`（`"bf16"` / `"fp16"` / `"none"`），訓練迴圈包 autocast：

```python
# train_predictor.py — 訓練 step
use_amp = cfg.amp_dtype in ("bf16", "fp16")
amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp_dtype)

with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
    logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
    loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
```
- bf16 不需 `GradScaler`（建議 A100 用 bf16）；fp16 才需 scaler，T4 較適用。
- tokenizer 已凍結且 `encode` 在 `no_grad` 下，autocast 只包 predictor forward/loss。

### 驗收
- 對照 fp32 vs bf16 的 step/sec 與峰值顯存。
- 確認 val_loss 曲線無發散（bf16 通常穩定）。

---

## 4. 值得做但非急迫（下一輪再排）

### 4-1. 回測加速：批次預測 + KV cache（#53 / #192）
`backtest.py:90-105` 對每個 symbol、每個 rebalance date **逐檔呼叫 `predict()`**，是回測最大的 wall-clock 瓶頸（symbols × dates 次序列解碼）。
- 借 #53 的 batch prediction：把同一 rebalance date 的所有 symbol context 疊成 batch 一次推論。
- 借 #192 的 KV cache：自迴歸解碼複用 KV，縮短每次 `pred_len` 步的成本。
- 收益：回測從「數十分鐘」級降到「數分鐘」，讓 autoresearch 每輪更快。

### 4-2. 機率排序選股（#321）
目前 `backtest.py:107` 用**點預測**排序：`pred["close"].iloc[-1] / ctx["close"].iloc[-1] - 1`，且 `sample_count=1, top_k=1`（近似 greedy）。
借 #321 的 `average_samples=False`，取得每條 Monte Carlo 路徑後，可改用：
- **上漲機率**（P(終值 > 現值)）排序，而非期望漲幅；
- 或加上**不確定性過濾**（路徑離散度過大者剔除）。
通常能改善風險調整後報酬（Sharpe）。需配合 `sample_count > 1`。

### 4-3. core model sampling/quantization bug 對照（#238 / #262 / #244）
這些在 `model/`（非 `finetune_tw/`），需確認本地 base 是否已含修復：
- #238：greedy decoding 把參數 `top_k` 當函式呼叫、BSQ `get_codebook_entry` unpack scalar、`require_grad` 拼錯（未真正凍結）。
- #262：`top_k_top_p_filtering` 提早 return 導致 top-p 被跳過（若 4-2 改用 top_p 取樣才會踩到）。
- #232 已 MERGED（`torch.topk` 修復）——確認本地已含。
> 動作：對 `model/kronos.py`、`model/module.py` 跑一次 diff 比對上游，把已 MERGED 的修復先同步進來。

### 4-4. fetcher 強化（#311 其餘借鏡）
- 友善代號映射層（如 `2330` → `2330.TW`、`TWII` → `^TWII`），降低 config/CLI 心智負擔。
- 交易時段/交易日過濾，剔除停牌或資料異常日。

---

## 5. 落實順序與相依

```
P0-1 dataset 洩漏修復 ─┐
P0-2 amount 代理值 ────┼─→ 重訓 tokenizer(Cell5) → 重訓 predictor(Cell6) → 回測(Cell7) → 記錄基準
P0-3 bf16 AMP ────────┘   (P0-2 改特徵分佈，必須重訓 tokenizer；P0-1/P0-3 只需重訓 predictor)
                                              │
                                              ▼
              4-1 回測加速 → 4-2 機率排序 → 4-3 core bug 同步 → 4-4 fetcher 強化
```

**關鍵相依**：P0-2 改動了輸入特徵分佈 → **tokenizer 必須重訓**（重跑 Cell 5+6）；P0-1 與 P0-3 只動 predictor 訓練 → 只需重跑 Cell 6+7。因此**三個 P0 一起改、一次重訓**最省事。

**每步驟前先 `git commit` 保存 config 與程式碼，方便回滾**（呼應 CLAUDE.md 的迭代守則）。

---

## 6. 驗收指標（沿用 CLAUDE.md 收斂條件）

| 指標 | 目標 | 本設計關注點 |
|------|------|------------|
| Sharpe | ≥ 1.5（vs ^TWII） | P0-1 修完應「下降但可信」，後續靠 4-2 拉回 |
| Annual Return | > 15%（2024-07-01 起） | — |
| Max Drawdown | < 20% | 4-2 不確定性過濾有助降低 |
| 連續改善 | < 1%（連 2 輪）視為收斂 | — |

> 重點認知：P0-1 修完，指標**很可能變差**——那不是退步，是把先前「洩漏造成的虛高」擠掉。**修復後的數字才是真正的起跑線**，後續所有改善都以此為基準。

---

## 7. 不採納 / 範圍外

- WebUI / Studio / MetaTrader / cTrader 等前端 PR（#281、#309、#292、#324…）——與台股訓練無關。
- 其他市場的完整腳本（美股 #310/#318、印度 #311）——**只借設計，不引入檔案**，避免 `finetune_tw` 範圍膨脹。
- 文件/typo PR（#251、#258、#291…）——無實質幫助。
