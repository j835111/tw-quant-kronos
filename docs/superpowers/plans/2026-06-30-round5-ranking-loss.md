# Round 5 Training Plan: Pretrained Restart + Auxiliary Ranking Loss

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train from pretrained Kronos-base (bypassing Round 0 local optimum) with auxiliary open-to-open@h5 IC ranking loss that directly optimizes the backtest signal.

**Architecture:**
- Pretrained restart: `NeoQuasar/Kronos-base` predictor (no fine-tuned weights), full parameter fine-tuning (no FPT freeze).
- Auxiliary ranking loss: every `ranking_loss_every_n_steps` CE steps, sample a cross-sectional batch (all available stocks on a random training date), compute IC loss between soft-predicted returns (via precomputed S1 oracle table) and actual realized open returns, then backprop. This interleaves CE and IC objectives without changing the main training data pipeline.
- S1 Oracle: empirically built from training data. For each S1 token ID (0–1023), stores the mean realized open-to-open return 5 days out for training windows that resolved to that S1 token. Frozen tokenizer makes this stable and precomputable once.
- Early stopping: open-to-open IC-IR@h5 (direct proxy for backtest Sharpe).

**Tech Stack:** PyTorch, KronosTokenizer (frozen, s1_bits=10 s2_bits=10), Kronos predictor (d_model=832), SQLite TWSE DB, existing `finetune_tw` infra.

## Global Constraints

- Python ≥ 3.10; PyTorch ≥ 2.0.
- Tokenizer is always frozen (`p.requires_grad_(False)` before training starts).
- Pretrained predictor: `NeoQuasar/Kronos-base` (no `hf_revision`).
- All paths relative to repo root `/workspace/Kronos/` on RunPod A40.
- `pred_len` must be `≥ ranking_loss_horizon + 1` (needs open[T+h+1]).
- Oracle is computed once at training start; never updated during training.
- Ranking loss is computed cross-sectionally: all stocks on one date per ranking batch, never mixing dates.
- Tests run offline (no HuggingFace downloads) using toy in-memory fixtures.

---

### Task 1: Config — Round 5 Pretrained Restart

**Files:**
- Create: `finetune_tw/configs/config_tw_daily_round5_a40.yaml`

**Interfaces:**
- Produces: `Config` object loadable by `Config.from_yaml()` with new fields `ranking_loss_alpha`, `ranking_loss_horizon`, `ranking_loss_every_n_steps`, `oracle_min_count`.

- [ ] **Step 1: Create config file**

```yaml
# RunPod A40 (48 GB) — Round 5: pretrained restart + auxiliary ranking loss
db_path: "/workspace/Kronos/finetune_tw/data/tw_stocks.db"
lookback_window: 90
predict_window: 10
max_context: 512
clip: 5.0
train_end_date: "2023-12-31"
val_end_date: "2024-06-30"

tokenizer_epochs: 12
basemodel_epochs: 20
batch_size: 256
save_steps: 500
log_interval: 50
tokenizer_lr: 0.0002
predictor_lr: 0.00005
adam_beta1: 0.9
adam_beta2: 0.95
adam_weight_decay: 0.1
num_workers: 4
persistent_workers: true
prefetch_factor: 2
train_steps_per_epoch: 1000
val_steps_per_epoch: 200
amp_dtype: "bf16"
enable_tf32: true
token_cache_enabled: true
token_cache_dtype: "int32"
seed: 42

early_stop_patience: 5
ic_val_symbols: 150
ic_val_dates: 40
ic_target_horizon: 5          # open-to-open IC-IR@h5 (matches backtest signal)
fpt_freeze: false             # full fine-tuning from pretrained

# Auxiliary ranking loss (new in Round 5)
ranking_loss_alpha: 0.1       # weight of IC loss relative to CE loss
ranking_loss_horizon: 5       # open[T+h+1]/open[T+1]-1 target horizon
ranking_loss_every_n_steps: 5 # run ranking batch every N CE steps
oracle_min_count: 20          # min samples per S1 token to include in oracle
cross_sectional_batch_size: 64 # stocks per cross-sectional ranking batch

# Pretrained restart (key change vs previous rounds)
pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
pretrained_predictor: "NeoQuasar/Kronos-base"
hf_revision: ""               # empty = load pretrained directly (no fine-tuned revision)
hf_tokenizer_revision: ""
hf_repo: "j835111/kronos-tw-finetune"
hf_revision_out: "round-5"
hf_checkpoint_revision_out: "checkpoints-round-5"
hf_checkpoint_keep_last_n: 3
exp_name: "tw_daily"
output_dir: "/workspace/Kronos/finetune_tw/outputs"

top_k: 10
hold_days: 5
pred_len: 11                  # max_hold + 1 = 5 + 1, then +5 for open[T+h+1] buffer → use 11
test_start_date: "2024-07-01"
benchmark_symbol: "^TWII"
```

- [ ] **Step 2: Add new fields to Config dataclass**

In `finetune_tw/config.py`, add to the `Config` dataclass:

```python
ranking_loss_every_n_steps: int = 5
oracle_min_count: int = 20
cross_sectional_batch_size: int = 64
```

(Fields `ranking_loss_alpha`, `ranking_loss_horizon` already exist in `Config`.)

- [ ] **Step 3: Verify config loads**

```bash
python3 -c "
from finetune_tw.config import Config
cfg = Config.from_yaml('finetune_tw/configs/config_tw_daily_round5_a40.yaml')
assert cfg.ranking_loss_alpha == 0.1
assert cfg.ic_target_horizon == 5
assert cfg.fpt_freeze == False
assert cfg.hf_revision == ''
print('Config OK')
"
```

Expected: `Config OK`

- [ ] **Step 4: Handle empty hf_revision in train_predictor.py**

In `train_predictor.py` around line 436, the current code does:

```python
if cfg.hf_revision:
    # download specific revision from HF
else:
    model = Kronos.from_pretrained(cfg.pretrained_predictor).to(device)
```

Verify the `else` branch already handles pretrained correctly. Since `hf_revision: ""` is falsy in Python, the `else` branch loads `NeoQuasar/Kronos-base` directly — confirm this is correct and add a `print` statement:

```python
else:
    model = Kronos.from_pretrained(cfg.pretrained_predictor).to(device)
    print(f"  Loaded pretrained predictor from {cfg.pretrained_predictor}")
```

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/configs/config_tw_daily_round5_a40.yaml finetune_tw/config.py finetune_tw/train_predictor.py
git commit -m "config: add Round 5 pretrained-restart + ranking-loss config"
```

---

### Task 2: S1 Oracle Table

**Files:**
- Create: `finetune_tw/score_oracle.py`
- Test: `tests/finetune_tw/test_score_oracle.py`

**Interfaces:**
- Produces: `build_s1_oracle(tokenizer, db_path, start, end, lookback, predict_window, horizon, clip, seed, min_count) -> torch.Tensor` shape `[V_s1]` (float32), where `V_s1 = 2 ** tokenizer.s1_bits`.
- Produces: `oracle_pred_score(s1_logits_at_h, oracle) -> torch.Tensor` shape `[B]`, differentiable.

- [ ] **Step 1: Write failing test**

```python
# tests/finetune_tw/test_score_oracle.py
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_fake_tokenizer(s1_bits=4, s2_bits=4):
    """Returns a mock tokenizer with encode() returning random token IDs."""
    tok = MagicMock()
    tok.s1_bits = s1_bits
    V_s1 = 2 ** s1_bits
    V_s2 = 2 ** s2_bits

    def fake_encode(x, half=False):
        B, T, _ = x.shape
        s1 = torch.randint(0, V_s1, (B, T))
        s2 = torch.randint(0, V_s2, (B, T))
        return s1, s2

    tok.encode = fake_encode
    return tok, V_s1


def _make_fake_dataset(n_samples=200, lookback=10, pred_len=6):
    """Returns list of (x_tensor [T, 6], stamp) with synthetic open prices."""
    T = lookback + pred_len
    samples = []
    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        opens = 100 * np.cumprod(1 + rng.normal(0, 0.01, T))
        x = np.zeros((T, 6), dtype=np.float32)
        x[:, 0] = opens.astype(np.float32)  # open channel
        x[:, 3] = opens.astype(np.float32)  # close channel (dummy)
        samples.append((torch.from_numpy(x), torch.zeros(T, 5)))
    return samples


def test_build_s1_oracle_shape_and_type():
    from finetune_tw.score_oracle import build_s1_oracle_from_samples
    tok, V_s1 = _make_fake_tokenizer(s1_bits=4)
    dataset = _make_fake_dataset(n_samples=300, lookback=10, pred_len=6)
    oracle = build_s1_oracle_from_samples(tok, dataset, lookback=10, horizon=5, min_count=5)
    assert oracle.shape == (V_s1,), f"Expected ({V_s1},), got {oracle.shape}"
    assert oracle.dtype == torch.float32
    assert torch.isfinite(oracle).all()


def test_oracle_pred_score_differentiable():
    from finetune_tw.score_oracle import oracle_pred_score
    V_s1 = 16
    oracle = torch.randn(V_s1)
    s1_logits = torch.randn(8, V_s1, requires_grad=True)
    scores = oracle_pred_score(s1_logits, oracle)
    assert scores.shape == (8,)
    scores.sum().backward()
    assert s1_logits.grad is not None
    assert not torch.all(s1_logits.grad == 0)


def test_oracle_tokens_with_few_samples_get_zero():
    from finetune_tw.score_oracle import build_s1_oracle_from_samples
    tok, V_s1 = _make_fake_tokenizer(s1_bits=4)
    # Only 1 sample → all tokens below min_count=20 → oracle should be zeros
    dataset = _make_fake_dataset(n_samples=1, lookback=10, pred_len=6)
    oracle = build_s1_oracle_from_samples(tok, dataset, lookback=10, horizon=5, min_count=20)
    assert oracle.abs().sum() == 0.0
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/finetune_tw/test_score_oracle.py -v
```

Expected: `ImportError: cannot import name 'build_s1_oracle_from_samples'`

- [ ] **Step 3: Implement `finetune_tw/score_oracle.py`**

```python
"""
S1 Oracle: empirical lookup table mapping S1 token IDs → mean open-to-open return.

The KronosTokenizer is frozen throughout predictor training. For each training
window, the S1 token at position `lookback - 1` (the last context token) is a
fixed function of the input OHLCV history. We build an empirical table:

    oracle[s1_id] = mean(realized_open_return_at_h)
                    over all training windows where the last S1 token == s1_id

This gives a differentiable predicted score during training:
    pred_score = softmax(s1_logits_at_h) @ oracle   (dot product, [B])

Gradient flows through the softmax, pushing the model to assign higher
probability to S1 tokens that historically correlate with higher open returns.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def build_s1_oracle_from_samples(
    tokenizer,
    dataset,
    lookback: int,
    horizon: int,
    min_count: int = 20,
) -> torch.Tensor:
    """Build oracle table from an iterable of (x_tensor [T, 6], stamp) pairs.

    Parameters
    ----------
    tokenizer : KronosTokenizer (frozen, eval mode)
        Must expose `s1_bits` attribute and `encode(x, half=False)`.
    dataset : iterable of (Tensor[T, 6], any)
        Each x has T = lookback + predict_window rows, open channel at col 0.
    lookback : int
        Number of context rows (the first `lookback` rows).
    horizon : int
        Target horizon h. Realized return = open[lookback+h] / open[lookback] - 1.
    min_count : int
        Tokens with fewer samples get oracle value 0.

    Returns
    -------
    oracle : Tensor[V_s1]  float32
        oracle[i] = mean realized return for windows with last-context S1 token == i.
        Tokens with count < min_count are set to 0.0.
    """
    V_s1 = 2 ** tokenizer.s1_bits
    buckets: dict[int, list[float]] = defaultdict(list)

    tokenizer_device = next(
        (p.device for p in tokenizer.parameters() if hasattr(p, "device")), torch.device("cpu")
    )

    for item in dataset:
        x, _stamp = item
        T = x.shape[0]
        if T < lookback + horizon + 1:
            continue

        opens = x[:, 0].float()
        denom = float(opens[lookback].item())
        if denom <= 0:
            continue
        realized_ret = float(opens[lookback + horizon].item()) / denom - 1.0
        if not np.isfinite(realized_ret):
            continue

        with torch.no_grad():
            x_in = x.unsqueeze(0).to(tokenizer_device)
            s1_ids, _ = tokenizer.encode(x_in, half=True)  # [1, T]
            last_s1 = int(s1_ids[0, lookback - 1].item())

        buckets[last_s1].append(realized_ret)

    oracle = torch.zeros(V_s1, dtype=torch.float32)
    for s1_id, rets in buckets.items():
        if len(rets) >= min_count:
            oracle[s1_id] = float(np.mean(rets))
    return oracle


def oracle_pred_score(
    s1_logits_at_h: torch.Tensor,
    oracle: torch.Tensor,
) -> torch.Tensor:
    """Differentiable predicted return score from S1 logits and oracle table.

    Parameters
    ----------
    s1_logits_at_h : Tensor[B, V_s1]
        S1 logits at the sequence position corresponding to horizon h.
    oracle : Tensor[V_s1]
        Precomputed oracle table (from build_s1_oracle_from_samples).

    Returns
    -------
    scores : Tensor[B]  (requires_grad if s1_logits_at_h does)
        scores[b] = sum_i softmax(s1_logits_at_h[b])[i] * oracle[i]
    """
    probs = F.softmax(s1_logits_at_h, dim=-1)  # [B, V_s1]
    return (probs * oracle.to(probs.device)).sum(dim=-1)  # [B]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/finetune_tw/test_score_oracle.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/score_oracle.py tests/finetune_tw/test_score_oracle.py
git commit -m "feat: S1 oracle table for differentiable ranking score"
```

---

### Task 3: Cross-Sectional Date Sampler

**Files:**
- Create: `finetune_tw/cross_sectional_dataset.py`
- Test: `tests/finetune_tw/test_cross_sectional_dataset.py`

**Interfaces:**
- Produces: `CrossSectionalDateSampler` class with `sample_date_batch(n_stocks, seed) -> dict` returning:
  - `"x"`: Tensor[N, T, 6] (normalized OHLCV context windows)
  - `"stamps"`: Tensor[N, T, 5]
  - `"actual_return_h"`: Tensor[N] realized open-to-open return at horizon h (float32)
  - `"date"`: the sampled training date (str)

- [ ] **Step 1: Write failing test**

```python
# tests/finetune_tw/test_cross_sectional_dataset.py
import torch
import numpy as np
import pytest


def _make_toy_db(tmp_path, n_syms=5, n_days=120):
    """Create a minimal SQLite DB with synthetic OHLCV data."""
    import sqlite3, pandas as pd

    db = tmp_path / "toy.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE stocks (symbol TEXT, date TEXT, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL, amount REAL)"
    )
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    for sym in [f"SYM{i:04d}.TW" for i in range(n_syms)]:
        price = 100.0
        for d in dates:
            price *= 1 + rng.normal(0, 0.01)
            conn.execute(
                "INSERT INTO stocks VALUES (?,?,?,?,?,?,?,?)",
                (sym, d, price, price * 1.01, price * 0.99, price, 1000.0, 1000.0 * price),
            )
    conn.commit()
    conn.close()
    return str(db)


def test_sample_date_batch_shapes(tmp_path):
    from finetune_tw.cross_sectional_dataset import CrossSectionalDateSampler

    db = _make_toy_db(tmp_path, n_syms=5, n_days=120)
    sampler = CrossSectionalDateSampler(
        db_path=db,
        lookback=20,
        horizon=5,
        start_date="2020-01-02",
        end_date="2020-10-01",
        clip=5.0,
        seed=42,
    )
    batch = sampler.sample_date_batch(n_stocks=3, seed=0)
    assert "x" in batch and "actual_return_h" in batch
    N = batch["x"].shape[0]
    assert N <= 3
    assert batch["x"].ndim == 3 and batch["x"].shape[2] == 6
    assert batch["actual_return_h"].shape == (N,)
    assert batch["actual_return_h"].dtype == torch.float32


def test_sample_date_batch_actual_returns_finite(tmp_path):
    from finetune_tw.cross_sectional_dataset import CrossSectionalDateSampler

    db = _make_toy_db(tmp_path, n_syms=5, n_days=120)
    sampler = CrossSectionalDateSampler(
        db_path=db,
        lookback=20,
        horizon=5,
        start_date="2020-01-02",
        end_date="2020-10-01",
        clip=5.0,
        seed=42,
    )
    for seed in range(5):
        batch = sampler.sample_date_batch(n_stocks=5, seed=seed)
        assert torch.isfinite(batch["actual_return_h"]).all()


def test_different_seeds_give_different_dates(tmp_path):
    from finetune_tw.cross_sectional_dataset import CrossSectionalDateSampler

    db = _make_toy_db(tmp_path, n_syms=5, n_days=120)
    sampler = CrossSectionalDateSampler(
        db_path=db,
        lookback=20,
        horizon=5,
        start_date="2020-01-02",
        end_date="2020-10-01",
        clip=5.0,
        seed=42,
    )
    dates = {sampler.sample_date_batch(5, seed=i)["date"] for i in range(20)}
    assert len(dates) > 1
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/finetune_tw/test_cross_sectional_dataset.py -v
```

Expected: `ImportError: cannot import name 'CrossSectionalDateSampler'`

- [ ] **Step 3: Implement `finetune_tw/cross_sectional_dataset.py`**

```python
"""Cross-sectional date sampler for auxiliary ranking loss computation.

Randomly samples a trading date from the training window, loads all available
stock contexts ending on that date (lookback rows), applies the same z-score
normalization as MultiStockDataset, and returns both the normalized context
tensors and the realized open-to-open return at horizon h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from finetune_tw.db import list_symbols, query_symbol

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


class CrossSectionalDateSampler:
    """Provides cross-sectional batches (all stocks on one date) for ranking loss.

    Parameters
    ----------
    db_path : str
    lookback : int  Number of context rows per sample (must match training lookback_window).
    horizon : int   h in open[T+h+1]/open[T+1]-1.
    start_date : str  Earliest possible signal date (inclusive).
    end_date : str    Latest possible signal date (inclusive).
    clip : float  Clipping for z-score normalization (matches training cfg.clip).
    seed : int    RNG seed for reproducible date sampling.
    benchmark_symbol : str  Excluded from universe.
    """

    def __init__(
        self,
        db_path: str,
        lookback: int,
        horizon: int,
        start_date: str,
        end_date: str,
        clip: float = 5.0,
        seed: int = 42,
        benchmark_symbol: str = "^TWII",
    ) -> None:
        self.db_path = db_path
        self.lookback = lookback
        self.horizon = horizon
        self.clip = clip
        self.benchmark_symbol = benchmark_symbol

        # Collect valid signal dates: business days in [start_date, end_date]
        self._dates: list[str] = (
            pd.bdate_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
        )
        self._rng = np.random.default_rng(seed)
        self._symbols: list[str] = [
            s for s in list_symbols(db_path) if s != benchmark_symbol
        ]

    def sample_date_batch(
        self,
        n_stocks: int,
        seed: int | None = None,
    ) -> dict:
        """Sample a random date and return up to n_stocks cross-sectional contexts.

        Returns a dict with keys:
          "x"               : Tensor[N, lookback, 6]  normalized context
          "stamps"          : Tensor[N, lookback, 5]  not used by ranking loss but included
          "actual_return_h" : Tensor[N]  realized open[T+h+1]/open[T+1]-1
          "date"            : str        the sampled date
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        date_str = rng.choice(self._dates)
        # Need lookback rows ending on date_str, plus horizon+1 future rows for realized return
        lookback_start = (
            pd.Timestamp(date_str) - pd.Timedelta(days=self.lookback * 3)
        ).strftime("%Y-%m-%d")
        future_end = (
            pd.Timestamp(date_str) + pd.Timedelta(days=(self.horizon + 1) * 3)
        ).strftime("%Y-%m-%d")

        sym_order = list(self._symbols)
        rng.shuffle(sym_order)

        xs, actual_rets = [], []
        for sym in sym_order:
            if len(xs) >= n_stocks:
                break
            df = query_symbol(self.db_path, sym, start=lookback_start, end=future_end)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Find the row index of date_str (signal date)
            signal_mask = df["date"] == pd.Timestamp(date_str)
            if not signal_mask.any():
                continue
            signal_idx = int(signal_mask.idxmax())

            # Need lookback rows ending at signal_idx (inclusive)
            if signal_idx < self.lookback - 1:
                continue
            ctx = df.iloc[signal_idx - self.lookback + 1 : signal_idx + 1]

            # Need horizon+1 future rows starting right after signal_idx
            future = df.iloc[signal_idx + 1 : signal_idx + self.horizon + 2]
            if len(future) < self.horizon + 1:
                continue

            open_t1 = float(future.iloc[0]["open"])
            open_th1 = float(future.iloc[self.horizon]["open"])
            if open_t1 <= 0:
                continue
            realized_ret = open_th1 / open_t1 - 1.0
            if not np.isfinite(realized_ret):
                continue

            arr = ctx[FEATURES].values.astype(np.float32)
            past = arr[: self.lookback]
            mean = past.mean(axis=0)
            std = past.std(axis=0) + 1e-5
            arr_norm = np.clip((arr - mean) / std, -self.clip, self.clip)

            xs.append(torch.from_numpy(arr_norm))
            actual_rets.append(realized_ret)

        if not xs:
            return {
                "x": torch.zeros(0, self.lookback, 6),
                "actual_return_h": torch.zeros(0),
                "date": date_str,
            }

        return {
            "x": torch.stack(xs),
            "actual_return_h": torch.tensor(actual_rets, dtype=torch.float32),
            "date": date_str,
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/finetune_tw/test_cross_sectional_dataset.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/cross_sectional_dataset.py tests/finetune_tw/test_cross_sectional_dataset.py
git commit -m "feat: CrossSectionalDateSampler for auxiliary ranking loss"
```

---

### Task 4: Wire Ranking Loss into Training Loop

**Files:**
- Modify: `finetune_tw/train_predictor.py`

**Interfaces:**
- Consumes: `build_s1_oracle_from_samples` (Task 2), `oracle_pred_score` (Task 2), `CrossSectionalDateSampler` (Task 3).
- The main training loop alternates: every `cfg.ranking_loss_every_n_steps` CE steps, run one cross-sectional ranking step and add IC loss × `cfg.ranking_loss_alpha` to the gradient.

- [ ] **Step 1: Write failing integration test**

```python
# tests/finetune_tw/test_ranking_loss_integration.py
import torch
import pytest


def test_combine_training_loss_with_real_ranking_loss():
    from finetune_tw.train_predictor import (
        differentiable_rank_ic_loss,
        _combine_training_loss,
    )
    # IC loss between two random score vectors
    pred_scores = torch.randn(10, requires_grad=True)
    actual_scores = torch.randn(10)
    ranking_loss = differentiable_rank_ic_loss(pred_scores, actual_scores)
    ce_loss = torch.tensor(3.5, requires_grad=True)
    total = _combine_training_loss(ce_loss, ranking_loss_alpha=0.1, ranking_loss=ranking_loss)
    total.backward()
    # Gradient must flow to pred_scores
    assert pred_scores.grad is not None


def test_differentiable_rank_ic_loss_perfect_agreement():
    from finetune_tw.train_predictor import differentiable_rank_ic_loss
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # Perfect positive correlation → negative IC → loss approaches -1
    loss = differentiable_rank_ic_loss(scores, scores.clone())
    assert loss.item() < -0.9


def test_differentiable_rank_ic_loss_constant_pred_returns_zero():
    from finetune_tw.train_predictor import differentiable_rank_ic_loss
    pred = torch.ones(5)
    actual = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    loss = differentiable_rank_ic_loss(pred, actual)
    # std of pred ≈ 0 → z_pred ≈ 0 → loss ≈ 0
    assert abs(loss.item()) < 0.01
```

- [ ] **Step 2: Run test to confirm**

```bash
pytest tests/finetune_tw/test_ranking_loss_integration.py -v
```

Expected: all 3 PASS (functions already exist; this just validates their behavior).

- [ ] **Step 3: Add `_run_cross_sectional_ranking_step` to `train_predictor.py`**

Add this function after `_validate_predictor`:

```python
def _run_cross_sectional_ranking_step(
    model,
    tokenizer,
    cross_sampler,
    oracle_s1: torch.Tensor,
    cfg,
    device,
    amp_enabled: bool,
    amp_dtype,
    horizon: int,
) -> torch.Tensor | None:
    """Sample one cross-sectional batch and return IC ranking loss (or None if skipped).

    The S1 logit at position (lookback_window - 1 + horizon) in the output sequence
    corresponds to predicting the token for day T+horizon.  oracle_pred_score converts
    that logit distribution to an expected return score, then IC loss measures whether
    predicted scores rank stocks similarly to actual realized returns.
    """
    from finetune_tw.score_oracle import oracle_pred_score

    batch = cross_sampler.sample_date_batch(
        n_stocks=cfg.cross_sectional_batch_size,
        seed=None,
    )
    if batch["x"].shape[0] < 3:
        return None  # too few stocks on this date — skip

    x = batch["x"].to(device)  # [N, lookback, 6]
    actual_ret = batch["actual_return_h"].to(device)  # [N]

    with torch.no_grad():
        token_s1, token_s2 = tokenizer.encode(x, half=True)  # [N, lookback]

    # Autoregressive: token_in is [:-1], output logit at pos k predicts token k+1
    # Horizon position in the output logits = (lookback - 1) + (horizon - 1)
    # = lookback + horizon - 2  (0-indexed)
    # But we're using only the context (lookback rows), so max logit pos = lookback - 2.
    # For the PREDICTOR model (not tokenizer), it needs to autoregressively generate
    # horizon steps ahead. We use decode_s1 to get the logit at the last context position
    # as a proxy signal — this is the "what token comes next" distribution which correlates
    # with the near-term directional move.
    #
    # NOTE: We use the LAST context position's logit (pos lookback-1) as the ranking
    # signal. This is the model's distribution over S1 tokens immediately after the
    # context window, which the oracle maps to expected returns.
    # For a future improvement, run 'horizon' autoregressive steps to get the h-step logit.

    stamps_in = _build_ranking_stamps(x, device)  # [N, lookback, 5]

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.bfloat16,
        enabled=amp_enabled,
    ):
        s1_logits_all, _ = model.decode_s1(
            token_s1,
            token_s2,
            stamp=stamps_in,
        )
        # s1_logits_all: [N, lookback, V_s1]; last position = token at position lookback
        s1_logits_at_last = s1_logits_all[:, -1, :]  # [N, V_s1]
        pred_scores = oracle_pred_score(s1_logits_at_last, oracle_s1)  # [N], requires_grad

    ranking_loss = differentiable_rank_ic_loss(pred_scores, actual_ret)
    return ranking_loss


def _build_ranking_stamps(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build a dummy timestamp tensor for ranking batches (all zeros except hour=9)."""
    N, T, _ = x.shape
    stamps = torch.zeros(N, T, 5, device=device)
    stamps[:, :, 1] = 9.0  # hour = 9 (market open), matches training convention
    return stamps
```

- [ ] **Step 4: Integrate into the training loop in `run_training`**

In `run_training`, after the tokenizer/model setup section (around line 430), add:

```python
# --- Ranking loss setup (Round 5+) ---
oracle_s1 = None
cross_sampler = None
ranking_loss_alpha = getattr(cfg, "ranking_loss_alpha", 0.0)
ranking_loss_every_n = getattr(cfg, "ranking_loss_every_n_steps", 5)
ranking_loss_horizon = getattr(cfg, "ranking_loss_horizon", 5)
if ranking_loss_alpha > 0.0:
    print(f"  Building S1 oracle (horizon={ranking_loss_horizon})...")
    from finetune_tw.score_oracle import build_s1_oracle_from_samples
    from finetune_tw.cross_sectional_dataset import CrossSectionalDateSampler
    # Build oracle from training dataset (MultiStockDataset already loaded above)
    oracle_s1 = build_s1_oracle_from_samples(
        tokenizer, train_ds, lookback=cfg.lookback_window,
        horizon=ranking_loss_horizon,
        min_count=getattr(cfg, "oracle_min_count", 20),
    ).to(device)
    n_nonzero = int((oracle_s1 != 0).sum().item())
    print(f"  Oracle built: {n_nonzero}/{oracle_s1.shape[0]} tokens with data.")
    cross_sampler = CrossSectionalDateSampler(
        db_path=cfg.db_path,
        lookback=cfg.lookback_window,
        horizon=ranking_loss_horizon,
        start_date="2015-01-01",
        end_date=cfg.train_end_date,
        clip=cfg.clip,
        seed=cfg.seed,
        benchmark_symbol=cfg.benchmark_symbol,
    )
```

Then in the inner training loop, replace:

```python
total_loss = _combine_training_loss(loss, cfg.ranking_loss_alpha)
```

with:

```python
ranking_loss = None
if (
    oracle_s1 is not None
    and cross_sampler is not None
    and global_step % ranking_loss_every_n == 0
):
    ranking_loss = _run_cross_sectional_ranking_step(
        model=model,
        tokenizer=tokenizer,
        cross_sampler=cross_sampler,
        oracle_s1=oracle_s1,
        cfg=cfg,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        horizon=ranking_loss_horizon,
    )
total_loss = _combine_training_loss(loss, ranking_loss_alpha, ranking_loss)
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/finetune_tw/ -v -k "ranking"
```

Expected: all ranking-related tests PASS.

- [ ] **Step 6: Commit**

```bash
git add finetune_tw/train_predictor.py tests/finetune_tw/test_ranking_loss_integration.py
git commit -m "feat: wire auxiliary cross-sectional ranking loss into training loop"
```

---

### Task 5: Training Run + Evaluation Guide

This task is a reference guide for executing the training on RunPod A40 and evaluating results.

**Pre-flight checklist before starting training:**

- [ ] **Step 1: Sync latest branch to RunPod**

```bash
# On local machine
git push origin research/round-5

# On RunPod (in /workspace/Kronos)
git fetch origin && git checkout research/round-5 && git pull origin research/round-5
```

- [ ] **Step 2: Confirm tokenizer exists (use Round 0's tokenizer)**

```bash
ls /workspace/Kronos/finetune_tw/outputs/tw_daily/tokenizer/best_model/
# Should show: config.json  model.safetensors
```

If missing, pull from HuggingFace:
```bash
python3 -c "
from finetune_tw.hf_utils import restore_best_model
from pathlib import Path
restore_best_model(
    Path('finetune_tw/outputs/tw_daily'),
    'j835111/kronos-tw-finetune',
    'tokenizer/best_model',
    'round-0',
)
"
```

- [ ] **Step 3: Delete stale token cache (config changed)**

```bash
rm -rf /workspace/Kronos/finetune_tw/outputs/tw_daily/token_cache/
```

- [ ] **Step 4: Start training**

```bash
cd /workspace/Kronos
python -m finetune_tw.train_predictor \
    --config finetune_tw/configs/config_tw_daily_round5_a40.yaml \
    2>&1 | tee /workspace/Kronos/finetune_tw/outputs/tw_daily/train_log_round5_stdout.txt
```

Expected early output:
```
  Loaded pretrained predictor from NeoQuasar/Kronos-base
  Building S1 oracle (horizon=5)...
  Oracle built: XXX/1024 tokens with data.
  [epoch 1 step 5] loss=X.XXXX
  [epoch 1 step 10] loss=X.XXXX ranking_loss=X.XXXX
```

- [ ] **Step 5: Monitor early stop signal**

After each epoch, look for:
```
  val_loss=X.XXXX  val_ic=X.XXXX  ic_ir_h5=X.XXXX
  -> new best ic_ir_h5=X.XXXX ...
```

If `ic_ir_h5` is consistently NaN or < 0.05 after 3 epochs, the ranking signal is too weak. In that case:
- Reduce `ic_target_horizon` to 1 in the config (`ic_target_horizon: 1`) and restart.
- Do NOT switch back to Round 0 as starting point.

- [ ] **Step 6: Evaluate with backtest_next_open**

After training completes (or at best checkpoint):

```bash
python -m finetune_tw.backtest_next_open \
    --config finetune_tw/configs/config_tw_daily_round5_a40.yaml \
    --model round5 \
    --hold_days_list 5 10 \
    2>&1 | tee /tmp/backtest_round5.txt
```

(You may need to add `round5` to the `choices` list in `backtest_next_open.py`'s `main()` and register it in `build_model_specs`.)

- [ ] **Step 7: Compare with Round 0 baseline**

Target comparison:

| 指標 | Round 0 baseline | Round 5 target |
|------|-----------------|----------------|
| open/open Sharpe (hold=5d) | 1.12 | > 1.12 (ideally > 1.30) |
| Annual Return | 38.6% | > 38.6% |
| MaxDD | 35% | < 35% |

Record results in `autoresearch/tw-evals/finetune-tw-results.tsv` and `docs/kronos-tw-round-history.md`.

- [ ] **Step 8: Commit results**

```bash
git add autoresearch/tw-evals/finetune-tw-results.tsv docs/kronos-tw-round-history.md
git commit -m "results: Round 5 backtest results (open/open Sharpe=X.XX)"
```

---

## Self-Review

**Spec coverage:**
- ✅ Pretrained restart (Task 1: `hf_revision: ""`)
- ✅ Auxiliary ranking loss targeting open-to-open@h5 (Task 2 oracle + Task 3 sampler + Task 4 integration)
- ✅ Differentiable score via oracle soft-argmax (Task 2 `oracle_pred_score`)
- ✅ IC-IR@h5 early stopping (Task 1: `ic_target_horizon: 5`)
- ✅ Full fine-tuning, no FPT (Task 1: `fpt_freeze: false`)
- ✅ Tests for each new component (Tasks 2, 3, 4)

**Placeholder scan:** None detected.

**Known limitation:** `_run_cross_sectional_ranking_step` uses the **last context position logit** as the ranking signal, not the true horizon-h logit (which would require running h autoregressive decoding steps per ranking batch). This is a practical approximation: the oracle maps `last-context-S1-logit → expected future return`, capturing the model's "next token" prediction quality as a proxy for multi-step ranking. A future improvement (Round 6) would run full autoregressive decoding for h steps to get the true h-step predicted score.
