# Kronos-TW 上游 PR 採納（P0 硬傷修復）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修掉 `finetune_tw/` 三個已對照程式碼確認存在的硬傷——正規化資料洩漏、`amount` 死特徵、predictor 訓練缺混合精度——讓下一輪回測指標可信且 Colab 迭代更快。

**Architecture:** 三個獨立但需「一次重訓」綁定的修復。P0-1 改 `dataset.py` 正規化只用 lookback 段；P0-2 改 `yfinance_fetcher.py` 用 `volume × 均價` 當成交額代理；P0-3 在 `config.py` 加 `amp_dtype` 欄位、`train_predictor.py` 抽出 `_resolve_amp` helper 並用 `torch.autocast` 包住 forward/loss。全程 TDD，pytest 在無 GPU/無網路下即可驗證（DB 用 tmp_path、fetcher 用 mock）。

**Tech Stack:** Python、PyTorch、pandas、numpy、pytest、yfinance（mock）、SQLite。

**來源依據：** 設計文件 `docs/superpowers/specs/2026-06-19-kronos-tw-upstream-pr-adoption-design.md`；上游 PR #263/#234（洩漏）、#311（成交額代理）、#288/#289（bf16）。

## Global Constraints

- 輸入特徵固定 6 維：`FEATURES = ["open","high","low","close","volume","amount"]`，對應模型 `d_in=6`。不得增減欄位順序。
- P0-2 改變輸入特徵分佈 → **tokenizer 必須重訓**（Colab Cell 5+6）。P0-1、P0-3 只動 predictor 訓練 → 只需重訓 Cell 6。三個 P0 一起改、一次重訓。
- 混合精度本計畫**只支援 `bf16` 與 `none`**。`fp16`（需 `GradScaler`）列為後續工作，不在本計畫範圍——避免無 scaler 的 fp16 靜默劣化訓練。
- 既有 fetcher 的 `amount` 語意必須一致為「成交金額（新台幣）」。`finmind`（`Trading_money`）與 `twse`（第 3 欄）已提供真實金額，**不得改動**；只有 yfinance 缺此欄才用代理。
- TDD：每個 code step 都先寫失敗測試。測試指令一律從 repo 根目錄執行。
- 每個 Task 結尾 commit；commit message 結尾加：`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

---

## File Structure

| 檔案 | 責任 | 動作 |
|------|------|------|
| `finetune_tw/dataset.py` | `MultiStockDataset` 正規化只用 lookback 段 | Modify |
| `tests/finetune_tw/test_dataset.py` | 新增「預測期不影響正規化統計」測試 | Modify |
| `finetune_tw/fetchers/yfinance_fetcher.py` | `amount` 改為 `volume × 均價` 代理 | Modify |
| `tests/finetune_tw/test_fetchers.py` | 改寫 amount 測試為代理值斷言 | Modify |
| `finetune_tw/config.py` | 新增 `amp_dtype` 欄位 | Modify |
| `finetune_tw/configs/config_tw_daily.yaml` | 新增 `amp_dtype: "bf16"` | Modify |
| `finetune_tw/train_predictor.py` | `_resolve_amp` helper + `torch.autocast` 包住 forward/loss | Modify |
| `tests/finetune_tw/test_train_predictor.py` | `_resolve_amp` 純單元測試 | Create |

---

## Task 1：P0-1 消除正規化資料洩漏

**Files:**
- Modify: `finetune_tw/dataset.py`（`__init__` 約 28 行；`__getitem__` 約 51-60 行）
- Test: `tests/finetune_tw/test_dataset.py`

**Interfaces:**
- Consumes: `init_db`, `upsert_prices`（`finetune_tw.db`）、`MultiStockDataset(db, lookback, predict, start, end, clip=5.0, seed=42)`。
- Produces: `MultiStockDataset` 行為不變（回傳 `(x_tensor[window,6], stamp_tensor[window,5])`），但正規化的 `mean/std` 只由前 `lookback_window` 根計算。新增實例屬性 `self.lookback_window: int`。

- [ ] **Step 1: 寫失敗測試**

在 `tests/finetune_tw/test_dataset.py` 檔案**末尾**新增：

```python
def test_dataset_normalization_excludes_predict_window(tmp_path):
    """正規化統計只能來自 lookback 段；預測期的極端值不得影響 mean/std。"""
    db = str(tmp_path / "leak.db")
    init_db(db)
    n = WINDOW  # 剛好一個 window = LOOKBACK + PRED + 1
    rng = np.random.default_rng(0)
    # lookback 段：中等分散；預測段：極端離群值（若洩漏會主宰統計量）
    close = np.concatenate([
        rng.uniform(100, 120, LOOKBACK),
        np.full(PRED + 1, 1e6),
    ])
    df = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(n, 1e6), "amount": np.zeros(n),
    })
    upsert_prices(db, "LEAK.TW", df)
    ds = MultiStockDataset(db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    x, _ = ds[0]
    lookback_close = x.numpy()[:LOOKBACK, 3]  # col 3 = close
    # 只有「統計量僅來自 lookback 段」時，lookback 段正規化後才會接近零均值。
    # 若洩漏，極端離群值會把 lookback 段全部壓成同號的大負值，均值遠離 0。
    assert abs(float(lookback_close.mean())) < 0.5
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_dataset.py::test_dataset_normalization_excludes_predict_window -v`
Expected: FAIL（assert 失敗，`lookback_close.mean()` 約 -0.8 量級，絕對值 > 0.5）

- [ ] **Step 3: 在 `__init__` 存下 lookback_window**

`finetune_tw/dataset.py` 將：

```python
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
```

改為：

```python
        self.window = lookback_window + predict_window + 1
        self.lookback_window = lookback_window
        self.clip = clip
        self.seed = seed
```

- [ ] **Step 4: 正規化只用 lookback 段**

`finetune_tw/dataset.py` `__getitem__` 將：

```python
        mean = x.mean(axis=0)
        std = x.std(axis=0) + 1e-5
        x = np.clip((x - mean) / std, -self.clip, self.clip)
```

改為：

```python
        past = x[: self.lookback_window]          # 只用歷史段算統計量，避免未來資訊洩漏
        mean = past.mean(axis=0)
        std = past.std(axis=0) + 1e-5
        x = np.clip((x - mean) / std, -self.clip, self.clip)
```

- [ ] **Step 5: 跑新測試 + 既有 dataset 測試確認全過**

Run: `pytest tests/finetune_tw/test_dataset.py -v -k "not tokenizer_train"`
Expected: PASS（含新測試與既有 `test_dataset_x_is_normalized` 等；`test_tokenizer_train_one_step` 無 GPU 會 skip，用 `-k` 排除以加速）

- [ ] **Step 6: Commit**

```bash
git add finetune_tw/dataset.py tests/finetune_tw/test_dataset.py
git commit -m "fix(finetune_tw): normalize using lookback window only to prevent leakage

Mean/std were computed over the full window (lookback+predict), leaking
future price statistics into training features. Restrict stats to the
lookback portion, matching upstream #263/#234.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2：P0-2 yfinance `amount` 用成交額代理

**Files:**
- Modify: `finetune_tw/fetchers/yfinance_fetcher.py`（`fetch_symbol` 約 44-52 行）
- Test: `tests/finetune_tw/test_fetchers.py`（改寫 `test_fetch_symbol_amount_is_zero`）

**Interfaces:**
- Consumes: `yf.Ticker(symbol).history(...)` 回傳含 `Open/High/Low/Close/Volume` 的 DataFrame（mock 提供）。
- Produces: `fetch_symbol` 回傳的 DataFrame 中 `amount = volume × (open+high+low+close)/4`，欄位順序與型別不變。

- [ ] **Step 1: 改寫 amount 測試（先讓它對新行為失敗）**

`tests/finetune_tw/test_fetchers.py` 將既有的：

```python
def test_fetch_symbol_amount_is_zero():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history(3)
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        df = fetch_symbol("2330.TW", start="2024-01-01")
    assert (df["amount"] == 0.0).all()
```

整段替換為：

```python
def test_fetch_symbol_amount_is_turnover_proxy():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history(3)
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        df = fetch_symbol("2330.TW", start="2024-01-01")
    # amount = volume * 均價, 均價 = (O+H+L+C)/4 = (100+101+99+100.5)/4 = 100.125
    expected = 1_000_000 * 100.125
    assert (df["amount"] - expected).abs().max() < 1e-6
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_fetchers.py::test_fetch_symbol_amount_is_turnover_proxy -v`
Expected: FAIL（目前 amount 寫死 0.0，與 `expected=100125000.0` 不符）

- [ ] **Step 3: 實作成交額代理**

`finetune_tw/fetchers/yfinance_fetcher.py` 將：

```python
        df = pd.DataFrame({
            "date": date_strs,
            "open": hist["Open"].values,
            "high": hist["High"].values,
            "low": hist["Low"].values,
            "close": hist["Close"].values,
            "volume": hist["Volume"].values,
            "amount": 0.0,
        })
```

改為：

```python
        avg_price = (hist["Open"] + hist["High"] + hist["Low"] + hist["Close"]) / 4.0
        df = pd.DataFrame({
            "date": date_strs,
            "open": hist["Open"].values,
            "high": hist["High"].values,
            "low": hist["Low"].values,
            "close": hist["Close"].values,
            "volume": hist["Volume"].values,
            "amount": (hist["Volume"] * avg_price).values,  # 成交額代理：股數 × 均價
        })
```

- [ ] **Step 4: 跑全部 fetcher 測試確認過**

Run: `pytest tests/finetune_tw/test_fetchers.py -v`
Expected: PASS（新 amount 測試通過；`test_fetch_symbol_returns_standard_columns` 仍驗證欄位順序不變；finmind/twse 測試不受影響）

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/fetchers/yfinance_fetcher.py tests/finetune_tw/test_fetchers.py
git commit -m "feat(finetune_tw): use turnover proxy for yfinance amount column

yfinance has no turnover field so amount was hardcoded 0, wasting 1/6 of
the model input. Compute amount = volume * mean(OHLC) as a proxy, matching
the approach in upstream #311. finmind/twse keep their real amount.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3：P0-3a `amp_dtype` config 欄位 + `_resolve_amp` helper

**Files:**
- Modify: `finetune_tw/config.py`（Training 區段）
- Modify: `finetune_tw/configs/config_tw_daily.yaml`
- Modify: `finetune_tw/train_predictor.py`（新增模組級 helper）
- Test: `tests/finetune_tw/test_train_predictor.py`（Create）

**Interfaces:**
- Consumes: `Config.amp_dtype: str`（預設 `"bf16"`）。
- Produces: `finetune_tw.train_predictor._resolve_amp(amp_dtype: str) -> tuple[bool, torch.dtype | None]`。回傳 `(enabled, dtype)`：`"bf16"`→`(True, torch.bfloat16)`；`"none"` 或其他→`(False, None)`。

- [ ] **Step 1: 寫失敗測試（純單元，無需 GPU）**

建立 `tests/finetune_tw/test_train_predictor.py`：

```python
import torch
from finetune_tw.train_predictor import _resolve_amp


def test_resolve_amp_bf16():
    enabled, dtype = _resolve_amp("bf16")
    assert enabled is True
    assert dtype == torch.bfloat16


def test_resolve_amp_none():
    enabled, dtype = _resolve_amp("none")
    assert enabled is False
    assert dtype is None


def test_resolve_amp_unknown_falls_back_to_disabled():
    enabled, dtype = _resolve_amp("fp16")  # 本計畫不支援 fp16，視為停用
    assert enabled is False
    assert dtype is None
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_train_predictor.py -v`
Expected: FAIL（`ImportError: cannot import name '_resolve_amp'`）

- [ ] **Step 3: 加 `amp_dtype` 到 Config**

`finetune_tw/config.py` 在 Training 區段的 `num_workers: int = 2` 之後新增一行：

```python
    num_workers: int = 2
    amp_dtype: str = "bf16"  # 混合精度: "bf16" | "none"（fp16 暫不支援）
    seed: int = 42
```

（即在 `num_workers` 與 `seed` 之間插入 `amp_dtype`。）

- [ ] **Step 4: 加 `amp_dtype` 到 yaml**

`finetune_tw/configs/config_tw_daily.yaml` 在 `num_workers: 2` 之後新增：

```yaml
num_workers: 2
amp_dtype: "bf16"
seed: 42
```

- [ ] **Step 5: 實作 `_resolve_amp` helper**

`finetune_tw/train_predictor.py` 在 import 區塊之後、`run_training` 之前新增模組級函式：

```python
def _resolve_amp(amp_dtype: str) -> tuple[bool, "torch.dtype | None"]:
    """Map config amp_dtype to (autocast_enabled, dtype). Only bf16 is supported."""
    if amp_dtype == "bf16":
        return True, torch.bfloat16
    return False, None
```

- [ ] **Step 6: 跑測試確認過**

Run: `pytest tests/finetune_tw/test_train_predictor.py -v`
Expected: PASS（三個測試全綠）

- [ ] **Step 7: Commit**

```bash
git add finetune_tw/config.py finetune_tw/configs/config_tw_daily.yaml finetune_tw/train_predictor.py tests/finetune_tw/test_train_predictor.py
git commit -m "feat(finetune_tw): add amp_dtype config and _resolve_amp helper

Introduce bf16 mixed-precision config (default bf16) with a testable
_resolve_amp mapping. Wiring into the training loop follows. Based on
upstream #288/#289.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4：P0-3b 訓練迴圈套用 `torch.autocast`

**Files:**
- Modify: `finetune_tw/train_predictor.py`（`run_training` 訓練迴圈，約 61-80 行；`_validate_predictor`，約 102-116 行）

**Interfaces:**
- Consumes: `_resolve_amp(cfg.amp_dtype)`（Task 3 產出）、`device`（`torch.device`）。
- Produces: 訓練與驗證的 `model(...) forward + compute_loss` 在 `torch.autocast(device_type="cuda", dtype=..., enabled=...)` 下執行；bf16 不需 `GradScaler`，`loss.backward()` 與 optimizer 流程不變。CPU/非 cuda 裝置 autocast 自動停用。

- [ ] **Step 1: 在 `run_training` 解析 AMP 設定**

`finetune_tw/train_predictor.py`，在 `model = Kronos.from_pretrained(cfg.pretrained_predictor).to(device)` 之後新增：

```python
    amp_enabled, amp_dtype = _resolve_amp(cfg.amp_dtype)
    amp_enabled = amp_enabled and device.type == "cuda"
```

- [ ] **Step 2: 訓練 forward/loss 包進 autocast**

將訓練迴圈中：

```python
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            optimizer.zero_grad()
            loss.backward()
```

改為：

```python
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            optimizer.zero_grad()
            loss.backward()
```

- [ ] **Step 3: 驗證迴圈也包 autocast（與訓練數值一致）**

`_validate_predictor` 簽名改為接收 `amp_enabled, amp_dtype`：

將：

```python
def _validate_predictor(model, tokenizer, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_x_stamp in loader:
            batch_x       = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)
            token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
            token_in  = [token_s1[:, :-1], token_s2[:, :-1]]
            token_out = [token_s1[:, 1:],  token_s2[:, 1:]]
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            total += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
    return total / count if count else 0.0
```

改為：

```python
def _validate_predictor(model, tokenizer, loader, device, amp_enabled=False, amp_dtype=None) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_x_stamp in loader:
            batch_x       = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)
            token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
            token_in  = [token_s1[:, :-1], token_s2[:, :-1]]
            token_out = [token_s1[:, 1:],  token_s2[:, 1:]]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            total += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
    return total / count if count else 0.0
```

- [ ] **Step 4: 更新 `_validate_predictor` 呼叫端**

將：

```python
        val_loss = _validate_predictor(model, tokenizer, val_loader, device)
```

改為：

```python
        val_loss = _validate_predictor(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)
```

- [ ] **Step 5: 語法/匯入健檢（無 GPU 環境）**

Run: `python -c "import finetune_tw.train_predictor as t; print('import ok'); print(t._resolve_amp('bf16'))"`
Expected: 印出 `import ok` 與 `(True, torch.bfloat16)`，無語法或匯入錯誤。

- [ ] **Step 6: 既有測試回歸（確認沒打壞 import 鏈）**

Run: `pytest tests/finetune_tw/ -v -k "not tokenizer_train"`
Expected: PASS（dataset/fetcher/db/backtest/train_predictor 測試全綠；需 GPU 的訓練 smoke 測試 skip）

- [ ] **Step 7: Commit**

```bash
git add finetune_tw/train_predictor.py
git commit -m "feat(finetune_tw): wire bf16 autocast into predictor train/val loops

Wrap forward+loss in torch.autocast (auto-disabled off-CUDA). bf16 needs
no GradScaler so backward/optimizer flow is unchanged. CLAUDE.md's 'AMP'
claim is now accurate.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5：重訓 + 回測基準（Colab，操作驗證）

> 這是**非程式碼**的人工驗證 Task：P0-2 改變了輸入特徵分佈，必須重訓 tokenizer 才能讓三個修復一起生效。不要 commit 任何程式碼；目的是產出「修復後的可信基準」。

**Files:** 無（在 Colab 執行訓練/回測）。

**Interfaces:**
- Consumes: 已合入 Task 1–4 的分支、`finetune_tw/configs/config_tw_daily.yaml`（含 `amp_dtype: "bf16"`）。
- Produces: `finetune_tw/outputs/tw_daily/backtest_result.png` 與終端機印出的 Strategy/Benchmark 指標，作為「修復後新起跑線」。

- [ ] **Step 1: 同步分支到 Colab 並重抓資料（amount 已改）**

在 `colab_setup.ipynb`：Cell 2 `git pull` 取得本計畫分支；Cell 4 用 `--update` 重抓/回填，確保 yfinance 來源的 `amount` 是新代理值。
驗證：抽查一檔，`amount` 量級應為百億級新台幣而非 0。

- [ ] **Step 2: 重訓 tokenizer（必須，因特徵分佈改變）**

執行 Cell 5：`python finetune_tw/train_tokenizer.py`
Expected: 正常收斂並存出 `outputs/tw_daily/tokenizer/best_model/`。

- [ ] **Step 3: 重訓 predictor（bf16 生效）**

執行 Cell 6：`python finetune_tw/train_predictor.py`
Expected: 訓練啟動；A100/支援 bf16 的 GPU 上 step/sec 較先前 fp32 提升、峰值顯存下降；val_loss 不發散。

- [ ] **Step 4: 回測並記錄新基準**

執行 Cell 7：`python finetune_tw/backtest.py`
Expected: 印出 Strategy 與 ^TWII 的 Annual Return / Sharpe / Max DD，並存出 `backtest_result.png`。

- [ ] **Step 5: 與修復前對照、記錄結論**

把「修復前 vs 修復後」的 Sharpe / Annual Return / Max DD 記在 PR 描述或 commit note。
**預期認知**：修復後 Sharpe 很可能**下降**——那是擠掉洩漏造成的虛高，修復後的數字才是後續 autoresearch 的真正基準。若指標反而提升，需檢查是否 P0-2 的 amount 代理帶來實質資訊增益（合理）或資料有誤。

---

## Self-Review

**1. Spec（設計文件）coverage：**
- P0-1 洩漏 → Task 1 ✅
- P0-2 amount 代理 → Task 2 ✅
- P0-3 bf16 → Task 3（config+helper）+ Task 4（接線）✅
- 「一起重訓」相依 → Task 5 ✅
- 設計第 4 節「值得做但非急迫」（回測加速 #53/#192、機率排序 #321、core bug #238/#262）**刻意不納入本計畫**——屬獨立子系統，依 writing-plans 範圍守則應另開計畫。已在此明列以免被視為遺漏。

**2. Placeholder scan：** 無 TBD/「適當處理」等佔位字樣；每個 code step 都附完整程式碼與確切指令、預期輸出。

**3. Type consistency：**
- `_resolve_amp(amp_dtype: str) -> (bool, torch.dtype | None)`：Task 3 定義、Task 4 解構為 `amp_enabled, amp_dtype` 使用，名稱一致。
- `_validate_predictor(..., amp_enabled=False, amp_dtype=None)`：Task 4 簽名與呼叫端參數順序一致。
- `self.lookback_window`：Task 1 於 `__init__` 定義、`__getitem__` 使用，一致。
- `MultiStockDataset` 回傳形狀 `(window,6)/(window,5)` 不變，既有測試續用。
