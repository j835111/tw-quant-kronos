# Kronos Embedding → XGBoost + LambdaRankIC (Round 6 / M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the pretrained `NeoQuasar/Kronos-base` predictor, extract its last-layer hidden state as a 512-dim embedding per `(date, symbol)`, and train an XGBoost ranking model on top of these embeddings with a LambdaRank-style objective targeting Spearman Rank-IC — replacing the "fine-tune Kronos directly" approach that failed across Rounds 1–5.

**Architecture:** Three new scripts in `finetune_tw/`: (1) `extract_embeddings.py` batches OHLCV windows through the frozen tokenizer+model to cache mean-pooled hidden states + realized open-to-open labels as parquet; (2) `lambdarank_ic.py` + `train_xgb_lambdarank.py` train XGBoost on those embeddings with a custom pairwise objective weighted by label-rank distance (approximating Spearman IC optimization, since LambdaRank generalizes to any target metric by substituting its gain term); (3) `backtest_xgb_embedding.py` wires XGBoost's predictions into the **existing, unmodified** `signals_to_holdings` / `build_next_open_portfolio_returns` / `compute_metrics` harness from `finetune_tw/backtest.py` and `backtest_next_open.py` so the result is directly comparable to Round 0's Sharpe 1.12 baseline.

**Tech Stack:** PyTorch (frozen Kronos forward pass), `xgboost` Python package (needs `pip install xgboost` — confirmed missing in this environment), `pandas`/`numpy`/`pyarrow` (parquet cache), pytest.

## Global Constraints

- Kronos weights (`tokenizer`, `model`) are **frozen** — never call `.backward()` or an optimizer step on them. Always wrap forward passes in `torch.no_grad()` and call `.eval()`.
- Embedding extraction must reuse `KronosPredictor.prepare_batch_inputs` for normalization (lookback-only mean/std + clip) — do not hand-roll normalization, to stay bit-identical with `predict()`/`predict_batch()`.
- The label/signal convention must match the existing deployed metric exactly: **open-to-open return**, `open[T+h+1]/open[T+1]-1`, computed the same way `backtest_next_open.py::compute_raw_signals_open` computes it. This is what Round 0's Sharpe 1.12 baseline was measured on, and the whole point of this round is a same-basis comparison.
- Default ranking horizon `h=5` (matches `cfg.hold_days=5` / `cfg.ranking_loss_horizon=5`, the only executable hold period found viable in Rounds 0–5).
- Reuse existing helpers — do not duplicate: `finetune_tw.backtest_data.load_symbol_history_frames`, `build_rebalance_inputs`; `finetune_tw.backtest.compute_metrics`, `rank_stocks`, `signals_to_holdings`, `load_predictor_from_spec`, `build_model_specs`; `finetune_tw.backtest_next_open.build_next_open_portfolio_returns`, `_build_signal_and_execution_dates`; `finetune_tw.ic_validation.rank_ic`, `mean_cross_sectional_ic`.
- Test convention: `tests/finetune_tw/test_<module>.py`, pytest, no network/HF downloads in unit tests (construct tiny in-process models, per `tests/finetune_tw/test_kronos_predictor_batch.py`'s `_make_predictor_stub` pattern).
- This environment has no CUDA and is missing `xgboost` (`ModuleNotFoundError`) — Task 1 unit tests must run on CPU with a tiny synthetic Kronos model; full-scale embedding extraction + XGBoost training runs on RunPod (per `finetune_tw/molab_train.py` / `docs/kronos-tw-round-history.md` conventions), not in this dev environment.

---

### Task 1: Frozen Kronos embedding extraction

**Files:**
- Create: `finetune_tw/extract_embeddings.py`
- Test: `tests/finetune_tw/test_extract_embeddings.py`

**Interfaces:**
- Produces: `extract_embeddings_batch(predictor: KronosPredictor, df_list: list[pd.DataFrame], x_timestamp_list: list[pd.Series]) -> np.ndarray` shape `(B, d_model)`.
- Produces: `build_embedding_dataset(cfg: Config, predictor: KronosPredictor, symbols: list[str], rebal_dates: pd.DatetimeIndex, horizon: int) -> pd.DataFrame` with columns `["date", "symbol", "emb_0"..."emb_{d_model-1}", "label"]`.
- Consumes: `finetune_tw.backtest_data.load_symbol_history_frames`, `build_rebalance_inputs` (both already exist, signatures shown below).
- Consumes: `KronosPredictor.prepare_batch_inputs(df_list, x_timestamp_list, y_timestamp_list, pred_len)` → `(x_batch, x_stamp_batch, y_stamp_batch, means, stds, y_index_list)` (existing method, `model/kronos.py:564`).
- Consumes: `predictor.tokenizer.encode(x_tensor, half=True) -> (s1_ids, s2_ids)` and `predictor.model.decode_s1(s1_ids, s2_ids, stamp) -> (s1_logits, context)` where `context` has shape `(B, seq_len, d_model)` (existing methods, `model/kronos.py:142` and `model/kronos.py:278`).

```python
# Reference signatures already in the codebase (do not modify):
# finetune_tw/backtest_data.py
def load_symbol_history_frames(db_path: str, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]: ...
def build_rebalance_inputs(history_frames: dict[str, pd.DataFrame], symbols: list[str], rebal_date: pd.Timestamp, lookback_window: int, pred_len: int) -> tuple[list[str], list[pd.DataFrame], list[pd.Series], list[pd.Series]]: ...
```

- [ ] **Step 1: Write the failing tests for `extract_embeddings_batch`**

Create `tests/finetune_tw/test_extract_embeddings.py`:

```python
import numpy as np
import pandas as pd
import torch

from model.kronos import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.extract_embeddings import extract_embeddings_batch


def _make_tiny_predictor() -> KronosPredictor:
    torch.manual_seed(0)
    tokenizer = KronosTokenizer(
        d_in=6, d_model=8, n_heads=2, ff_dim=16,
        n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=2, s2_bits=2, beta=1e-2, gamma0=1.0, gamma=1.0, zeta=1.0,
        group_size=1,
    )
    model = Kronos(
        s1_bits=2, s2_bits=2, n_layers=1, d_model=8, n_heads=2, ff_dim=16,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=False,
    )
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=64)
    return predictor


def _make_df(offset: float, n: int = 20) -> pd.DataFrame:
    idx = np.arange(n, dtype=np.float32)
    return pd.DataFrame({
        "open": 10.0 + offset + idx * 0.1,
        "high": 10.5 + offset + idx * 0.1,
        "low": 9.5 + offset + idx * 0.1,
        "close": 10.2 + offset + idx * 0.1,
        "volume": 100.0 + idx,
        "amount": 1000.0 + idx * 10,
    })


def test_extract_embeddings_batch_shape():
    predictor = _make_tiny_predictor()
    df_list = [_make_df(0.0), _make_df(5.0), _make_df(-3.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    embeddings = extract_embeddings_batch(predictor, df_list, [x_ts, x_ts, x_ts])
    assert embeddings.shape == (3, 8)  # (batch, d_model)
    assert np.isfinite(embeddings).all()


def test_extract_embeddings_batch_is_deterministic_in_eval_mode():
    predictor = _make_tiny_predictor()
    df_list = [_make_df(0.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    first = extract_embeddings_batch(predictor, df_list, [x_ts])
    second = extract_embeddings_batch(predictor, df_list, [x_ts])
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_extract_embeddings_batch_distinguishes_different_inputs():
    predictor = _make_tiny_predictor()
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    embeddings = extract_embeddings_batch(predictor, [_make_df(0.0), _make_df(50.0)], [x_ts, x_ts])
    assert not np.allclose(embeddings[0], embeddings[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/finetune_tw/test_extract_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finetune_tw.extract_embeddings'`

- [ ] **Step 3: Implement `extract_embeddings_batch` and `build_embedding_dataset`**

Create `finetune_tw/extract_embeddings.py`:

```python
"""Extract frozen Kronos last-layer hidden states as embeddings for XGBoost ranking (Round 6 / M1).

Kronos is never updated here — this only runs forward passes under torch.no_grad().
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from finetune_tw.backtest import build_model_specs, load_predictor_from_spec
from finetune_tw.backtest_data import build_rebalance_inputs, load_symbol_history_frames
from finetune_tw.config import Config
from finetune_tw.db import list_symbols

BATCH_SIZE = 64


def extract_embeddings_batch(
    predictor,
    df_list: list[pd.DataFrame],
    x_timestamp_list: list[pd.Series],
) -> np.ndarray:
    """Mean-pool the frozen Kronos transformer's last-layer hidden state over the lookback window.

    Reuses KronosPredictor.prepare_batch_inputs for normalization (lookback-only mean/std + clip,
    identical to predict()/predict_batch()), then tokenizer.encode + model.decode_s1 to obtain the
    per-timestep context (B, seq_len, d_model), mean-pooled over seq_len -> (B, d_model).
    """
    if not df_list:
        return np.empty((0, predictor.model.d_model), dtype=np.float32)

    dummy_y_ts = [pd.Series(pd.bdate_range(x_ts.iloc[-1] + pd.Timedelta(days=1), periods=1))
                  for x_ts in x_timestamp_list]

    x_batch, x_stamp_batch, _, _, _, _ = predictor.prepare_batch_inputs(
        df_list=df_list,
        x_timestamp_list=x_timestamp_list,
        y_timestamp_list=dummy_y_ts,
        pred_len=1,
    )

    device = predictor.device
    x_tensor = torch.from_numpy(x_batch).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp_batch).to(device)

    with torch.no_grad():
        s1_ids, s2_ids = predictor.tokenizer.encode(x_tensor, half=True)
        _, context = predictor.model.decode_s1(s1_ids, s2_ids, x_stamp_tensor)
        pooled = context.mean(dim=1)  # (B, d_model) mean pooling over the lookback sequence

    return pooled.cpu().numpy().astype(np.float32)


def _realized_open_to_open_labels(
    price_frames: dict[str, pd.DataFrame],
    symbols: list[str],
    rebal_date: pd.Timestamp,
    horizon: int,
) -> dict[str, float]:
    """label[sym] = open[t+horizon+1] / open[t+1] - 1, matching backtest_next_open's signal exactly."""
    labels: dict[str, float] = {}
    for sym in symbols:
        frame = price_frames.get(sym)
        if frame is None:
            continue
        future = frame.loc[frame.index > rebal_date, "open"]
        if len(future) <= horizon:
            continue
        t1 = future.iloc[0]
        th = future.iloc[horizon]
        if t1 == 0:
            continue
        labels[sym] = float(th / t1 - 1.0)
    return labels


def build_embedding_dataset(
    cfg: Config,
    predictor,
    symbols: list[str],
    rebal_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    """One row per (date, symbol): embedding + realized open-to-open label at `horizon` days."""
    if len(rebal_dates) == 0:
        return pd.DataFrame()

    preload_start = (
        rebal_dates.min() - pd.Timedelta(days=cfg.lookback_window * 2)
    ).strftime("%Y-%m-%d")
    preload_end = (
        rebal_dates.max() + pd.Timedelta(days=horizon * 3)
    ).strftime("%Y-%m-%d")
    history_frames = load_symbol_history_frames(cfg.db_path, symbols, start=preload_start, end=preload_end)

    rows: list[dict] = []
    for i, rebal_date in enumerate(rebal_dates):
        cutoff = rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)
        recent = {sym: frame.loc[frame.index >= cutoff] for sym, frame in history_frames.items()}
        batch_syms, batch_dfs, batch_xts, _ = build_rebalance_inputs(
            recent, symbols, rebal_date, cfg.lookback_window, pred_len=1,
        )
        labels = _realized_open_to_open_labels(history_frames, batch_syms, rebal_date, horizon)

        keep_idx = [j for j, sym in enumerate(batch_syms) if sym in labels]
        if not keep_idx:
            continue
        kept_syms = [batch_syms[j] for j in keep_idx]
        kept_dfs = [batch_dfs[j] for j in keep_idx]
        kept_xts = [batch_xts[j] for j in keep_idx]

        for b in range(0, len(kept_syms), BATCH_SIZE):
            sub_syms = kept_syms[b:b + BATCH_SIZE]
            sub_dfs = kept_dfs[b:b + BATCH_SIZE]
            sub_xts = kept_xts[b:b + BATCH_SIZE]
            embeddings = extract_embeddings_batch(predictor, sub_dfs, sub_xts)
            for sym, emb in zip(sub_syms, embeddings):
                row = {"date": rebal_date.strftime("%Y-%m-%d"), "symbol": sym, "label": labels[sym]}
                row.update({f"emb_{k}": float(v) for k, v in enumerate(emb)})
                rows.append(row)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(rebal_dates)}] {rebal_date.date()}: {len(keep_idx)} symbols")
            sys.stdout.flush()

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--model", default="pretrained", choices=list(build_model_specs(Config()).keys()))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    specs = build_model_specs(cfg)
    predictor = load_predictor_from_spec(specs[args.model], cfg)

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    rebal_dates = pd.bdate_range(args.start, args.end)

    df = build_embedding_dataset(cfg, predictor, symbols, rebal_dates, args.horizon)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/finetune_tw/test_extract_embeddings.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/extract_embeddings.py tests/finetune_tw/test_extract_embeddings.py
git commit -m "feat(round6): add frozen Kronos embedding extraction (M1 step 1)"
```

---

### Task 2: LambdaRankIC objective (pure, testable module)

**Files:**
- Create: `finetune_tw/lambdarank_ic.py`
- Test: `tests/finetune_tw/test_lambdarank_ic.py`

**Interfaces:**
- Produces: `lambdarank_ic_grad_hess(preds: np.ndarray, labels: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]` — gradient/hessian for **one group** (one date's cross-section).
- Produces: `lambdarank_ic_objective(group_sizes: list[int], sigma: float = 1.0) -> Callable[[np.ndarray, "xgboost.DMatrix"], tuple[np.ndarray, np.ndarray]]` — returns an XGBoost-compatible `obj(preds, dtrain)` closure that slices `preds`/`dtrain.get_label()` per group and concatenates each group's `lambdarank_ic_grad_hess` output.
- Consumes: nothing new (pure numpy).

**Design note (документed assumption):** This implements the standard LambdaRank pairwise-logistic gradient, with the NDCG "gain" term replaced by `|rank(label_i) - rank(label_j)|` — a legitimate substitution since LambdaRank generalizes to any target metric by swapping in that metric's pairwise sensitivity (here, Spearman IC's sensitivity to a rank swap is linear in the rank distance of the swapped pair). This is our own derivation for Rank-IC, not a verbatim transcription of arXiv:2605.00501 (not read in full) — flag this in code comments so a future pass can reconcile with the paper's exact Equation 5 if higher fidelity is needed.

```python
# finetune_tw/lambdarank_ic.py
"""LambdaRank-style pairwise objective targeting Spearman Rank-IC, for XGBoost custom obj=.

Standard LambdaRank pairwise-logistic gradient, with the usual |ΔNDCG| gain term replaced by
the label-rank distance |rank(y_i) - rank(y_j)| (Spearman IC's sensitivity to swapping a pair's
predicted order is linear in that rank distance). This is our derivation for Rank-IC, not a
verbatim copy of arXiv:2605.00501 Eq. 5 — reconcile against the paper later if needed.
"""
from __future__ import annotations

import numpy as np


def _dense_rank(values: np.ndarray) -> np.ndarray:
    """1-based ascending rank, ties broken by stable sort order (matches pandas .rank(method='first'))."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def lambdarank_ic_grad_hess(
    preds: np.ndarray,
    labels: np.ndarray,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient/hessian for one cross-sectional group (one trading date)."""
    n = len(preds)
    grad = np.zeros(n, dtype=np.float64)
    hess = np.zeros(n, dtype=np.float64)
    if n < 2:
        return grad, hess

    label_ranks = _dense_rank(labels)

    # i should outrank j whenever label_i > label_j; vectorized over all pairs in the group.
    pred_diff = preds[:, None] - preds[None, :]          # s_i - s_j
    label_diff = labels[:, None] - labels[None, :]       # y_i - y_j
    rank_dist = np.abs(label_ranks[:, None] - label_ranks[None, :])

    pair_mask = label_diff > 0                            # only pairs where i should outrank j
    rho = 1.0 / (1.0 + np.exp(sigma * pred_diff))          # sigmoid(-sigma * pred_diff)
    lam = sigma * rho * rank_dist * pair_mask              # magnitude, zero outside mask
    hess_pair = (sigma ** 2) * rho * (1.0 - rho) * rank_dist * pair_mask

    # i is pushed up (negative grad), j is pushed down (positive grad).
    grad += -lam.sum(axis=1) + lam.sum(axis=0)
    hess += hess_pair.sum(axis=1) + hess_pair.sum(axis=0)

    hess = np.maximum(hess, 1e-6)  # XGBoost requires strictly positive hessian
    return grad, hess


def lambdarank_ic_objective(group_sizes: list[int], sigma: float = 1.0):
    """Return an XGBoost-compatible obj(preds, dtrain) -> (grad, hess) for the whole training set."""
    boundaries = np.cumsum([0] + list(group_sizes))

    def _obj(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label()
        grad = np.zeros_like(preds, dtype=np.float64)
        hess = np.zeros_like(preds, dtype=np.float64)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            g, h = lambdarank_ic_grad_hess(preds[start:end], labels[start:end], sigma=sigma)
            grad[start:end] = g
            hess[start:end] = h
        return grad, hess

    return _obj
```

- [ ] **Step 1: Write the failing tests**

Create `tests/finetune_tw/test_lambdarank_ic.py`:

```python
import numpy as np

from finetune_tw.lambdarank_ic import lambdarank_ic_grad_hess, lambdarank_ic_objective


def test_gradient_sign_pushes_misranked_pair_toward_correct_order():
    # labels: sample 0 is best (3), sample 2 is worst (1). preds are fully reversed.
    labels = np.array([3.0, 2.0, 1.0])
    preds = np.array([1.0, 2.0, 3.0])
    grad, hess = lambdarank_ic_grad_hess(preds, labels)

    assert grad[0] < 0    # best label, lowest pred -> must be pushed up
    assert grad[2] > 0    # worst label, highest pred -> must be pushed down
    assert (hess > 0).all()


def test_gradient_is_zero_for_perfectly_ranked_group():
    labels = np.array([3.0, 2.0, 1.0])
    preds = np.array([30.0, 20.0, 10.0])  # already in perfect descending order matching labels
    grad, _ = lambdarank_ic_grad_hess(preds, labels, sigma=1.0)
    # rho ~ 0 for confidently-correct pairs -> gradient magnitude near zero
    assert np.allclose(grad, 0.0, atol=1e-3)


def test_single_sample_group_returns_zero_gradient():
    grad, hess = lambdarank_ic_grad_hess(np.array([1.0]), np.array([1.0]))
    assert grad.shape == (1,)
    assert np.allclose(grad, 0.0)
    assert np.allclose(hess, 0.0)


class _FakeDMatrix:
    def __init__(self, labels):
        self._labels = labels

    def get_label(self):
        return self._labels


def test_objective_respects_group_boundaries():
    # 2 groups: [3.0, 2.0, 1.0] (2 samples) and [1.0, 5.0] (2 samples, mixed with group 1's labels)
    labels = np.array([3.0, 1.0, 1.0, 5.0])
    preds = np.array([1.0, 2.0, 2.0, 1.0])
    obj = lambdarank_ic_objective(group_sizes=[2, 2])
    grad, hess = obj(preds, _FakeDMatrix(labels))

    assert grad.shape == (4,)
    # group 1 (indices 0,1): label 3 > label 1 -> sample 0 (pred=1, lower) should be pushed up
    assert grad[0] < 0
    # group 2 (indices 2,3): label 1 < label 5 -> sample 3 (pred=1, lower) should be pushed up
    assert grad[3] < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/finetune_tw/test_lambdarank_ic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finetune_tw.lambdarank_ic'`

- [ ] **Step 3: Implement `finetune_tw/lambdarank_ic.py`**

Use the exact code shown above under Task 2's interface block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/finetune_tw/test_lambdarank_ic.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/lambdarank_ic.py tests/finetune_tw/test_lambdarank_ic.py
git commit -m "feat(round6): add LambdaRankIC pairwise objective (M1 step 2)"
```

---

### Task 3: XGBoost training script

**Files:**
- Create: `finetune_tw/train_xgb_lambdarank.py`
- Test: `tests/finetune_tw/test_train_xgb_lambdarank.py`

**Interfaces:**
- Produces: `build_group_sizes(df: pd.DataFrame, date_col: str = "date") -> list[int]` — group sizes in the row order of `df` sorted by `date_col` (stable), matching `xgboost.DMatrix.set_group` semantics.
- Produces: `rank_ic_eval_metric(preds: np.ndarray, dtrain, group_sizes: list[int]) -> float` — mean cross-sectional Rank-IC across groups, using `finetune_tw.ic_validation.rank_ic` (existing, do not reimplement).
- Consumes: `finetune_tw.lambdarank_ic.lambdarank_ic_objective` (Task 2).
- Consumes: `finetune_tw.ic_validation.rank_ic` (existing, `finetune_tw/ic_validation.py:13`).

```python
# finetune_tw/train_xgb_lambdarank.py
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

from finetune_tw.ic_validation import rank_ic
from finetune_tw.lambdarank_ic import lambdarank_ic_objective

EMBEDDING_PREFIX = "emb_"


def build_group_sizes(df: pd.DataFrame, date_col: str = "date") -> list[int]:
    """Row order MUST already be sorted by date_col before calling this (caller's responsibility)."""
    return df.groupby(date_col, sort=False).size().tolist()


def rank_ic_eval_metric(preds: np.ndarray, dtrain, group_sizes: list[int]) -> float:
    labels = dtrain.get_label()
    boundaries = np.cumsum([0] + list(group_sizes))
    ics = [
        rank_ic(preds[start:end], labels[start:end])
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics)) if ics else float("nan")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith(EMBEDDING_PREFIX)],
                  key=lambda c: int(c[len(EMBEDDING_PREFIX):]))


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
    params: dict | None = None,
) -> xgb.Booster:
    train_df = train_df.sort_values("date", kind="stable").reset_index(drop=True)
    val_df = val_df.sort_values("date", kind="stable").reset_index(drop=True)

    feat_cols = _feature_columns(train_df)
    train_groups = build_group_sizes(train_df)
    val_groups = build_group_sizes(val_df)

    dtrain = xgb.DMatrix(train_df[feat_cols].values, label=train_df["label"].values)
    dtrain.set_group(train_groups)
    dval = xgb.DMatrix(val_df[feat_cols].values, label=val_df["label"].values)
    dval.set_group(val_groups)

    obj = lambdarank_ic_objective(train_groups, sigma=1.0)

    def feval(preds, dmat):
        return "rank_ic", -rank_ic_eval_metric(preds, dmat, val_groups)  # XGBoost minimizes eval metric

    default_params = {"max_depth": 4, "eta": 0.05, "tree_method": "hist"}
    booster = xgb.train(
        {**default_params, **(params or {})},
        dtrain,
        num_boost_round=num_boost_round,
        obj=obj,
        evals=[(dval, "val")],
        feval=feval,
        maximize=False,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10,
    )
    return booster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Parquet from extract_embeddings.py")
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True, help="Output path for the trained booster (.json)")
    parser.add_argument("--num_boost_round", type=int, default=200)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    booster = train(train_df, val_df, args.num_boost_round, args.early_stopping_rounds)
    booster.save_model(args.out)
    print(f"Saved -> {args.out}  (best_iteration={booster.best_iteration})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1: Write the failing tests**

Create `tests/finetune_tw/test_train_xgb_lambdarank.py`:

```python
import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

from finetune_tw.train_xgb_lambdarank import build_group_sizes, rank_ic_eval_metric, train


def _make_synthetic_df(n_dates=6, n_symbols=20, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = f"2024-01-{d + 1:02d}"
        true_factor = rng.normal(size=n_symbols)
        for s in range(n_symbols):
            emb = rng.normal(size=8)
            emb[0] += true_factor[s]  # emb_0 carries signal correlated with the label
            label = true_factor[s] + rng.normal(scale=0.1)
            row = {"date": date, "symbol": f"S{s}", "label": label}
            row.update({f"emb_{k}": float(v) for k, v in enumerate(emb)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_group_sizes_matches_date_row_counts():
    df = _make_synthetic_df(n_dates=3, n_symbols=5)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    assert build_group_sizes(df) == [5, 5, 5]


def test_train_improves_rank_ic_over_untrained_baseline():
    train_df = _make_synthetic_df(n_dates=10, n_symbols=30, seed=1)
    val_df = _make_synthetic_df(n_dates=4, n_symbols=30, seed=2)

    booster = train(train_df, val_df, num_boost_round=50, early_stopping_rounds=10)

    feat_cols = [c for c in val_df.columns if c.startswith("emb_")]
    val_sorted = val_df.sort_values("date", kind="stable").reset_index(drop=True)
    dval = xgb.DMatrix(val_sorted[feat_cols].values)
    preds = booster.predict(dval)
    val_groups = build_group_sizes(val_sorted)

    class _Labeled:
        def get_label(self):
            return val_sorted["label"].values

    trained_ic = rank_ic_eval_metric(preds, _Labeled(), val_groups)
    assert trained_ic > 0.2  # emb_0 is strongly correlated with label by construction
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/finetune_tw/test_train_xgb_lambdarank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finetune_tw.train_xgb_lambdarank'` (and/or xgboost missing — install first, see Step 3note)

- [ ] **Step 3: Install xgboost, then implement `finetune_tw/train_xgb_lambdarank.py`**

Run: `pip install xgboost`

Then create the file with the exact code shown above under Task 3's interface block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/finetune_tw/test_train_xgb_lambdarank.py -v`
Expected: PASS (2 tests). If `test_train_improves_rank_ic_over_untrained_baseline` is flaky, raise `num_boost_round` or lower the IC threshold slightly — the synthetic data is constructed with a strong true signal so this should pass reliably.

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/train_xgb_lambdarank.py tests/finetune_tw/test_train_xgb_lambdarank.py
git commit -m "feat(round6): add XGBoost LambdaRankIC training script (M1 step 3)"
```

---

### Task 4: Wire XGBoost predictions into the existing backtest harness

**Files:**
- Create: `finetune_tw/backtest_xgb_embedding.py`
- Test: `tests/finetune_tw/test_backtest_xgb_embedding.py`

**Interfaces:**
- Produces: `xgb_signals_to_raw_preds(xgb_preds_by_date: dict[str, dict[str, float]], hold_days: int) -> dict[str, dict[str, pd.Series]]` — adapts scalar XGBoost predictions into the `raw_preds` format `signals_to_holdings` already expects (`finetune_tw/backtest.py:134`), by repeating each scalar `hold_days` times so `ret.iloc[hold_days - 1]` (the exact indexing `signals_to_holdings` performs) resolves to the XGBoost score.
- Consumes (unmodified, existing): `finetune_tw.backtest.signals_to_holdings`, `compute_metrics`; `finetune_tw.backtest_next_open._build_signal_and_execution_dates`, `build_next_open_portfolio_returns`, `_load_price_frames`, `_load_trading_calendar`.
- Consumes (Task 1/3): `finetune_tw.extract_embeddings.extract_embeddings_batch`; a saved XGBoost booster via `xgb.Booster().load_model(path)`.

```python
# finetune_tw/backtest_xgb_embedding.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import xgboost as xgb

from finetune_tw.backtest import (
    build_model_specs, compute_metrics, load_predictor_from_spec,
    rank_stocks, signals_to_holdings,
)
from finetune_tw.backtest_data import build_rebalance_inputs, load_symbol_history_frames
from finetune_tw.backtest_next_open import (
    _build_signal_and_execution_dates, _load_price_frames, _load_trading_calendar,
    build_next_open_portfolio_returns,
)
from finetune_tw.config import Config
from finetune_tw.db import list_symbols
from finetune_tw.extract_embeddings import extract_embeddings_batch

BATCH_SIZE = 64


def xgb_signals_to_raw_preds(
    xgb_preds_by_date: dict[str, dict[str, float]],
    hold_days: int,
) -> dict[str, dict[str, pd.Series]]:
    """Repeat each scalar prediction hold_days times so signals_to_holdings' ret.iloc[hold_days-1]
    indexing resolves to the XGBoost score, regardless of which hold_days variant is evaluated."""
    raw_preds: dict[str, dict[str, pd.Series]] = {}
    for date_str, sym_scores in xgb_preds_by_date.items():
        raw_preds[date_str] = {
            sym: pd.Series([score] * hold_days) for sym, score in sym_scores.items()
        }
    return raw_preds


def compute_xgb_signals(
    predictor,
    booster: xgb.Booster,
    cfg: Config,
    rebal_dates: pd.DatetimeIndex,
    symbols: list[str],
) -> dict[str, dict[str, float]]:
    preload_start = (rebal_dates.min() - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
    preload_end = rebal_dates.max().strftime("%Y-%m-%d")
    history_frames = load_symbol_history_frames(cfg.db_path, symbols, start=preload_start, end=preload_end)

    signals: dict[str, dict[str, float]] = {}
    for i, rebal_date in enumerate(rebal_dates):
        cutoff = rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)
        recent = {sym: frame.loc[frame.index >= cutoff] for sym, frame in history_frames.items()}
        batch_syms, batch_dfs, batch_xts, _ = build_rebalance_inputs(
            recent, symbols, rebal_date, cfg.lookback_window, pred_len=1,
        )

        date_scores: dict[str, float] = {}
        with torch.no_grad():
            for b in range(0, len(batch_syms), BATCH_SIZE):
                sub_syms = batch_syms[b:b + BATCH_SIZE]
                embeddings = extract_embeddings_batch(predictor, batch_dfs[b:b + BATCH_SIZE], batch_xts[b:b + BATCH_SIZE])
                dmat = xgb.DMatrix(embeddings)
                preds = booster.predict(dmat)
                for sym, pred in zip(sub_syms, preds):
                    date_scores[sym] = float(pred)

        signals[rebal_date.strftime("%Y-%m-%d")] = date_scores
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i + 1}/{len(rebal_dates)}] {rebal_date.date()}: {len(date_scores)} signals")
            sys.stdout.flush()

    return signals


def run_backtest_xgb_embedding(cfg: Config, model_key: str, xgb_model_path: str, hold_days_list: list[int], top_k: int) -> Path:
    specs = build_model_specs(cfg)
    predictor = load_predictor_from_spec(specs[model_key], cfg)
    booster = xgb.Booster()
    booster.load_model(xgb_model_path)

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    test_end = str(pd.Timestamp.today().date())
    trading_dates = _load_trading_calendar(cfg, test_end)

    variant_schedules = {hd: _build_signal_and_execution_dates(trading_dates, hold_days=hd) for hd in hold_days_list}
    all_signal_dates = sorted({d for dates, _ in variant_schedules.values() for d in dates})
    signal_dates = pd.DatetimeIndex(all_signal_dates)

    price_frames = _load_price_frames(cfg, symbols, test_end)
    xgb_preds_by_date = compute_xgb_signals(predictor, booster, cfg, signal_dates, symbols)
    del predictor
    torch.cuda.empty_cache()

    out_dir = Path(cfg.output_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    hold_variants: dict[str, dict] = {}
    for hd in hold_days_list:
        variant_signal_dates, variant_execution_dates = variant_schedules[hd]
        raw_preds = xgb_signals_to_raw_preds(xgb_preds_by_date, hd)
        holdings = signals_to_holdings(raw_preds, variant_signal_dates, hd, top_k, cfg.min_signal_threshold)
        _, daily_returns = build_next_open_portfolio_returns(
            price_frames=price_frames, holdings_sequence=holdings,
            execution_dates=variant_execution_dates, trading_dates=trading_dates,
        )
        metrics = compute_metrics(daily_returns)
        hold_variants[str(hd)] = {
            "dates": [d.strftime("%Y-%m-%d") for d in daily_returns.index],
            "daily_returns": daily_returns.tolist(),
            "metrics": metrics,
        }
        print(f"  top_k={top_k} hold={hd}d — Ann:{metrics['annualised_return']:.2%} "
              f"Sharpe:{metrics['sharpe']:.2f} DD:{metrics['max_drawdown']:.2%}")

    out_path = out_dir / "backtest_returns_xgb_embedding_next_open.json"
    out_path.write_text(json.dumps({"model_key": model_key, "top_k": top_k, "hold_variants": hold_variants}, indent=2))
    print(f"\nSaved -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--model", default="pretrained")
    parser.add_argument("--xgb_model", required=True)
    parser.add_argument("--hold_days_list", type=int, nargs="+", default=[5])
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    run_backtest_xgb_embedding(cfg, args.model, args.xgb_model, args.hold_days_list, args.top_k)


if __name__ == "__main__":
    main()
```

- [ ] **Step 1: Write the failing test**

Create `tests/finetune_tw/test_backtest_xgb_embedding.py`:

```python
import pandas as pd

from finetune_tw.backtest import rank_stocks, signals_to_holdings
from finetune_tw.backtest_xgb_embedding import xgb_signals_to_raw_preds


def test_xgb_signals_to_raw_preds_plugs_into_signals_to_holdings():
    xgb_preds_by_date = {
        "2024-01-02": {"A": 0.05, "B": 0.02, "C": -0.01, "D": 0.10},
    }
    raw_preds = xgb_signals_to_raw_preds(xgb_preds_by_date, hold_days=5)

    # Every symbol's series must be long enough for iloc[hold_days - 1] and equal the raw score.
    for sym, series in raw_preds["2024-01-02"].items():
        assert len(series) == 5
        assert series.iloc[4] == xgb_preds_by_date["2024-01-02"][sym]

    holdings = signals_to_holdings(raw_preds, pd.DatetimeIndex(["2024-01-02"]), hold_days=5, top_k=2)
    assert holdings == [{"D", "A"}]  # top 2 by score: D=0.10, A=0.05


def test_xgb_signals_to_raw_preds_is_hold_days_invariant():
    xgb_preds_by_date = {"2024-01-02": {"A": 0.05, "B": 0.02}}
    for hold_days in (3, 5, 10):
        raw_preds = xgb_signals_to_raw_preds(xgb_preds_by_date, hold_days=hold_days)
        assert raw_preds["2024-01-02"]["A"].iloc[hold_days - 1] == 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/finetune_tw/test_backtest_xgb_embedding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finetune_tw.backtest_xgb_embedding'`

- [ ] **Step 3: Implement `finetune_tw/backtest_xgb_embedding.py`**

Use the exact code shown above under Task 4's interface block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/finetune_tw/test_backtest_xgb_embedding.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/backtest_xgb_embedding.py tests/finetune_tw/test_backtest_xgb_embedding.py
git commit -m "feat(round6): wire XGBoost embedding signal into existing backtest harness (M1 step 4)"
```

---

## Additions from `autoresearch/improve-260701-1512/` (scanned 2026-07-02)

That research directory (`research-findings.md`, `improvement-plan.md`, `summary.md`) has 8 insights behind this M1 approach. Two are genuinely **integrable** into the pipeline already built in Tasks 1–4 — both are explicitly flagged as open ablations in `improvement-plan.md`'s own "DECISION NEEDED" section, not new speculation:

- **Insight #3's raw-feature fallback** ("純 hidden state vs hidden state + raw features（MA, momentum, volume ratio）？建議先試純 hidden state，如果效果弱再加 raw features") → **Task 5** below adds MA/momentum/volume-ratio features alongside the embedding, additive and backward-compatible.
- **Insight #8** (arXiv:2509.05801, arXiv:2511.15324 — early transformer layers encode local AR/trend/level structure, deep layers encode dispersion/change-point signal; hidden states differ meaningfully by depth) → **Task 6** below makes the pooling layer selectable instead of hardcoding "last layer only," as a configurable ablation.

**Explicitly NOT integrated, and why:**

- **Insight #4 (MoFO optimizer)**, **Insight #5 (L2-SP regularization)**, **Insight #6 (SSPT continued pretraining)**, **Insight #7 (Finetuner's Fallacy continued pretraining)**, and `summary.md`'s "M2: pretrained restart + close-to-close IC-IR@h1 early stopping (無 ranking loss)" are all **fine-tuning-based** — they modify Kronos's weights. M1's entire premise (per `improvement-plan.md`'s own "戰略轉向": arXiv:2511.18578 shows fine-tuning pretrained TSFMs for financial return prediction fails systematically, confirmed empirically by Rounds 1–5 in `docs/kronos-tw-round-history.md`) is to **stop fine-tuning Kronos and freeze it**. Bolting a fine-tuning regularizer onto a frozen-Kronos pipeline is incoherent — there's nothing to regularize. These remain valid ideas for a **separate, later round** that revisits fine-tuning (e.g., a "Round 7: pretrained restart + close IC-IR@h1 + MoFO/L2-SP" branch), not additions to this M1 plan.

---

### Task 5: Raw technical features alongside the Kronos embedding

**Files:**
- Modify: `finetune_tw/extract_embeddings.py`
- Modify: `finetune_tw/train_xgb_lambdarank.py`
- Modify: `finetune_tw/backtest_xgb_embedding.py`
- Test: `tests/finetune_tw/test_extract_embeddings.py` (add cases)

**Interfaces:**
- Produces: `compute_technical_features(df: pd.DataFrame) -> dict[str, float]` with keys `feat_ma5_dist`, `feat_ma20_dist`, `feat_momentum_10`, `feat_volume_ratio`, computed from the same lookback-window `df` that `extract_embeddings_batch` already receives (no new data loading).
- Modifies: `build_embedding_dataset`'s per-row dict to also include `compute_technical_features(df)`'s output, so parquet rows have `emb_0..emb_{d_model-1}` **and** `feat_ma5_dist`, `feat_ma20_dist`, `feat_momentum_10`, `feat_volume_ratio` columns.
- Modifies: `train_xgb_lambdarank._feature_columns` to also pick up `feat_` prefixed columns (currently only `emb_`).
- Modifies: `backtest_xgb_embedding.compute_xgb_signals` to compute the same `feat_*` columns per symbol/date at inference time, in the same column order as training, so the booster's `DMatrix` at backtest time has an identical feature layout to what it was trained on.

```python
# finetune_tw/extract_embeddings.py — add this function
def compute_technical_features(df: pd.DataFrame) -> dict[str, float]:
    """Raw technical features as a fallback/complement to the pure Kronos embedding, per
    improvement-plan.md's open decision point ("純 hidden state vs hidden state + raw features").
    Computed from the same lookback-window df already passed to extract_embeddings_batch."""
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    last_close = float(close[-1])

    ma5 = float(close[-5:].mean()) if len(close) >= 5 else float(close.mean())
    ma20 = float(close[-20:].mean()) if len(close) >= 20 else float(close.mean())
    momentum_10 = float(last_close / close[-11] - 1.0) if len(close) > 10 and close[-11] != 0 else 0.0
    recent_vol_mean = float(volume[-20:].mean()) if len(volume) >= 20 else float(volume.mean())

    return {
        "feat_ma5_dist": float(last_close / ma5 - 1.0) if ma5 != 0 else 0.0,
        "feat_ma20_dist": float(last_close / ma20 - 1.0) if ma20 != 0 else 0.0,
        "feat_momentum_10": momentum_10,
        "feat_volume_ratio": float(volume[-1] / recent_vol_mean) if recent_vol_mean != 0 else 1.0,
    }
```

```python
# finetune_tw/extract_embeddings.py — build_embedding_dataset's row-building loop, replace the
# existing `for sym, emb in zip(sub_syms, embeddings): ...` block with:
for sym, emb, ctx_df in zip(sub_syms, embeddings, sub_dfs):
    row = {"date": rebal_date.strftime("%Y-%m-%d"), "symbol": sym, "label": labels[sym]}
    row.update({f"emb_{k}": float(v) for k, v in enumerate(emb)})
    row.update(compute_technical_features(ctx_df))
    rows.append(row)
```

```python
# finetune_tw/train_xgb_lambdarank.py — replace _feature_columns with:
_TECH_FEATURE_COLUMNS = ["feat_ma5_dist", "feat_ma20_dist", "feat_momentum_10", "feat_volume_ratio"]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    emb_cols = sorted([c for c in df.columns if c.startswith(EMBEDDING_PREFIX)],
                       key=lambda c: int(c[len(EMBEDDING_PREFIX):]))
    tech_cols = [c for c in _TECH_FEATURE_COLUMNS if c in df.columns]
    return emb_cols + tech_cols
```

```python
# finetune_tw/backtest_xgb_embedding.py — in compute_xgb_signals, replace the embeddings/DMatrix
# construction inside the batch loop with:
from finetune_tw.extract_embeddings import compute_technical_features
from finetune_tw.train_xgb_lambdarank import _TECH_FEATURE_COLUMNS

# ... inside the `for b in range(0, len(batch_syms), BATCH_SIZE):` loop:
sub_dfs = batch_dfs[b:b + BATCH_SIZE]
embeddings = extract_embeddings_batch(predictor, sub_dfs, batch_xts[b:b + BATCH_SIZE])
tech_feats = np.array([[compute_technical_features(df)[c] for c in _TECH_FEATURE_COLUMNS]
                       for df in sub_dfs], dtype=np.float32)
features = np.concatenate([embeddings, tech_feats], axis=1)
dmat = xgb.DMatrix(features)
preds = booster.predict(dmat)
for sym, pred in zip(sub_syms, preds):
    date_scores[sym] = float(pred)
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/finetune_tw/test_extract_embeddings.py`:

```python
from finetune_tw.extract_embeddings import compute_technical_features


def test_compute_technical_features_matches_hand_calculation():
    n = 25
    idx = np.arange(n, dtype=np.float64)
    close = 100.0 + idx  # 100, 101, ..., 124
    df = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": np.full(n, 200.0), "amount": np.full(n, 20000.0),
    })
    # bump the last day's volume so feat_volume_ratio is not exactly 1.0
    df.loc[df.index[-1], "volume"] = 400.0

    feats = compute_technical_features(df)

    last_close = close[-1]                       # 124.0
    expected_ma5 = close[-5:].mean()              # mean(120..124) = 122.0
    expected_ma20 = close[-20:].mean()            # mean(105..124) = 114.5
    expected_momentum_10 = last_close / close[-11] - 1.0  # 124/114 - 1
    expected_vol_mean = np.concatenate([np.full(19, 200.0), [400.0]]).mean()
    expected_vol_ratio = 400.0 / expected_vol_mean

    assert feats["feat_ma5_dist"] == pytest.approx(last_close / expected_ma5 - 1.0)
    assert feats["feat_ma20_dist"] == pytest.approx(last_close / expected_ma20 - 1.0)
    assert feats["feat_momentum_10"] == pytest.approx(expected_momentum_10)
    assert feats["feat_volume_ratio"] == pytest.approx(expected_vol_ratio)


def test_compute_technical_features_handles_short_history():
    n = 3
    df = pd.DataFrame({
        "open": [10.0, 11.0, 12.0], "high": [10.5, 11.5, 12.5],
        "low": [9.5, 10.5, 11.5], "close": [10.0, 11.0, 12.0],
        "volume": [100.0, 100.0, 100.0], "amount": [1000.0, 1000.0, 1000.0],
    })
    feats = compute_technical_features(df)  # must not raise IndexError with < 5/20/11 rows
    assert set(feats) == {"feat_ma5_dist", "feat_ma20_dist", "feat_momentum_10", "feat_volume_ratio"}
    assert all(np.isfinite(v) for v in feats.values())
```

Add `import pytest` to the top of `tests/finetune_tw/test_extract_embeddings.py` if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/finetune_tw/test_extract_embeddings.py -v -k technical_features`
Expected: FAIL with `ImportError: cannot import name 'compute_technical_features'`

- [ ] **Step 3: Implement `compute_technical_features` and wire it through the three files**

Use the exact code shown above (all four code blocks under Task 5's interface section).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/finetune_tw/test_extract_embeddings.py tests/finetune_tw/test_train_xgb_lambdarank.py tests/finetune_tw/test_backtest_xgb_embedding.py -v`
Expected: PASS (all tests, including the 2 new ones plus all previously-passing tests in these three files — this step touches all three files so the full trio must be re-verified together)

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/extract_embeddings.py finetune_tw/train_xgb_lambdarank.py finetune_tw/backtest_xgb_embedding.py tests/finetune_tw/test_extract_embeddings.py
git commit -m "feat(round6): add raw technical features alongside Kronos embedding (M1 step 5)"
```

---

### Task 6: Configurable transformer-layer selection for embedding pooling

**Files:**
- Modify: `finetune_tw/extract_embeddings.py`
- Test: `tests/finetune_tw/test_extract_embeddings.py` (add cases)

**Interfaces:**
- Modifies: `extract_embeddings_batch(predictor, df_list, x_timestamp_list, layer_indices: list[int] | None = None) -> np.ndarray`. Default `layer_indices=None` preserves the exact current behavior (final-layer `context` from `model.decode_s1`, mean-pooled) — this keeps every already-committed call site and test working unchanged. When `layer_indices` is provided, output shape is `(B, len(layer_indices) * d_model)`: the transformer's forward prefix (embedding, temporal embedding, token dropout, per-layer blocks) is replicated manually using the model's existing public submodules (`model.embedding`, `model.time_emb`, `model.token_drop`, `model.transformer`) — no changes to `model/kronos.py` — capturing the mean-pooled output after each requested layer index and concatenating in `layer_indices` order.

```python
# finetune_tw/extract_embeddings.py — replace the body of extract_embeddings_batch's
# `with torch.no_grad():` block with:
    with torch.no_grad():
        s1_ids, s2_ids = predictor.tokenizer.encode(x_tensor, half=True)
        model = predictor.model

        if layer_indices is None:
            _, context = model.decode_s1(s1_ids, s2_ids, x_stamp_tensor)
            pooled = context.mean(dim=1)
        else:
            x = model.embedding([s1_ids, s2_ids])
            x = x + model.time_emb(x_stamp_tensor)
            x = model.token_drop(x)
            layer_outputs = []
            for i, layer in enumerate(model.transformer):
                x = layer(x)
                if i in layer_indices:
                    layer_outputs.append(x.mean(dim=1))
            pooled = torch.cat(layer_outputs, dim=1)

    return pooled.cpu().numpy().astype(np.float32)
```

Update the function signature line to add the new parameter:

```python
def extract_embeddings_batch(
    predictor,
    df_list: list[pd.DataFrame],
    x_timestamp_list: list[pd.Series],
    layer_indices: list[int] | None = None,
) -> np.ndarray:
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/finetune_tw/test_extract_embeddings.py`:

```python
def _make_tiny_predictor_multi_layer(n_layers: int = 3) -> KronosPredictor:
    torch.manual_seed(0)
    tokenizer = KronosTokenizer(
        d_in=6, d_model=8, n_heads=2, ff_dim=16,
        n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=2, s2_bits=2, beta=1e-2, gamma0=1.0, gamma=1.0, zeta=1.0,
        group_size=1,
    )
    model = Kronos(
        s1_bits=2, s2_bits=2, n_layers=n_layers, d_model=8, n_heads=2, ff_dim=16,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=False,
    )
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device="cpu", max_context=64)


def test_extract_embeddings_batch_default_layer_indices_matches_old_shape():
    predictor = _make_tiny_predictor_multi_layer(n_layers=3)
    df_list = [_make_df(0.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    embeddings = extract_embeddings_batch(predictor, df_list, [x_ts])
    assert embeddings.shape == (1, 8)  # unchanged: (batch, d_model), backward compatible


def test_extract_embeddings_batch_multi_layer_concatenates_selected_layers():
    predictor = _make_tiny_predictor_multi_layer(n_layers=3)
    df_list = [_make_df(0.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=20))

    single_layer = extract_embeddings_batch(predictor, df_list, [x_ts], layer_indices=[0])
    two_layers = extract_embeddings_batch(predictor, df_list, [x_ts], layer_indices=[0, 2])

    assert single_layer.shape == (1, 8)
    assert two_layers.shape == (1, 16)  # 2 layers * d_model=8
    np.testing.assert_allclose(two_layers[:, :8], single_layer, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/finetune_tw/test_extract_embeddings.py -v -k layer`
Expected: FAIL with `TypeError: extract_embeddings_batch() got an unexpected keyword argument 'layer_indices'`

- [ ] **Step 3: Implement the `layer_indices` parameter**

Use the exact code shown above under Task 6's interface section.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/finetune_tw/test_extract_embeddings.py -v`
Expected: PASS (all tests in the file, old and new — this is the same file modified by Task 5, so run the whole file, not just the new cases)

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/extract_embeddings.py tests/finetune_tw/test_extract_embeddings.py
git commit -m "feat(round6): add configurable transformer-layer selection for embedding pooling (M1 step 6)"
```

---

### Task 7: Run end-to-end and record Round 6 results

**Files:**
- Modify: `docs/kronos-tw-round-history.md` (append "Round 6" section, following the existing per-round format)
- Modify: `autoresearch/tw-evals/finetune-tw-results.tsv` (append one row, matching existing columns)

This task has no unit tests — it is a real training/backtest run and a documentation update.

**Local CPU vs RunPod GPU:** Unlike Rounds 0–5, M1 never backpropagates through Kronos (frozen forward passes only) and XGBoost trains natively on CPU in minutes — per `improvement-plan.md`'s own framing ("訓練在 CPU 上幾分鐘可完成") and `summary.md` ("立即可做（不需 GPU）"). This step does **not** require RunPod; it can run on any CPU-only machine with the DB and internet access to download `NeoQuasar/Kronos-base`. GPU only speeds up the batched embedding-extraction forward passes, it isn't required for correctness.

**Config file:** Use `finetune_tw/configs/config_tw_daily.yaml`, not `config_tw_daily_rtx6000.yaml` — the rtx6000 config hardcodes a RunPod workspace path (`db_path: /workspace/Kronos/...`) and points `pretrained_predictor` at `j835111/kronos-tw-finetune@round-0` (Round 0's *fine-tuned* checkpoint), not raw `NeoQuasar/Kronos-base`. M1 needs the actual frozen pretrained base model, which is `config_tw_daily.yaml`'s default (`pretrained_predictor: "NeoQuasar/Kronos-base"`, `db_path: "finetune_tw/data/tw_stocks.db"`, a relative path).

- [ ] **Step 1: Ensure the DB is present**

`finetune_tw/data/` is gitignored, so a fresh clone or worktree won't have `tw_stocks.db`. If running in a worktree that doesn't have it, symlink or copy it from wherever it already exists on this machine, e.g. `ln -s /mnt/d/project/Kronos/finetune_tw/data finetune_tw/data`. If running on RunPod instead, follow `docs/kronos-tw-round-history.md`'s RunPod conventions via the `using-runpodctl` skill (push this branch, pull it on the pod, `pip install xgboost`, download the DB there).

Task 5 (raw technical features) and Task 6 (`layer_indices` selection) are both live by this point. The baseline run below uses their defaults — Task 5's features are always computed and included (no flag needed), and Task 6's `layer_indices=None` (final layer only) is `extract_embeddings.py`'s default. If the baseline result in Step 5 undershoots Round 0, the cheapest next experiment (no code changes needed) is re-running Step 2 with a modified `--out` path after hand-editing `layer_indices` in `extract_embeddings.py`'s `main()` to sweep e.g. `[0, cfg_n_layers // 2, cfg_n_layers - 1]` — do this only if the baseline needs it, not preemptively.

- [ ] **Step 2: Extract embeddings for train/val/test windows**

**Known bottleneck (measured 2026-07-02 on an A40 pod): the per-date extraction loop in `build_embedding_dataset` is CPU-bound, not GPU-bound.** `nvidia-smi` showed 0% GPU utilization with a single Python process pegged at 100% of one core (of 96 available) — the per-date pandas slicing in `build_rebalance_inputs` is single-threaded and never lets the GPU forward pass become the bottleneck. Sequential single-process extraction over the full 2015-2023 history was estimated at ~54h on CPU-only and ~5.5h on a single GPU process. Splitting the date range into N independent CLI invocations (no code changes — `extract_embeddings.py`'s `--start`/`--end`/`--out` args already support this) run in parallel as separate OS processes pushes GPU utilization to 100% (measured: 8 concurrent processes, 32GB/46GB VRAM on an A40, ~8x wall-clock speedup, near-linear). Use `scripts/run_round6_parallel.sh` (already written) instead of two plain sequential CLI calls:

```bash
bash scripts/run_round6_parallel.sh
```

That script runs Steps 2–4 together: 8-way parallel train extraction (`2015-01-01`→`2023-12-31` split into 8 contiguous ranges) → merge to `/root/embeddings_train.parquet`, 4-way parallel val extraction (`2024-01-01`→`2024-06-30` split into 4 ranges) → merge to `/root/embeddings_val.parquet`, then Steps 3–4 below unchanged. It assumes it's run from `/root/Kronos` with the DB already at `finetune_tw/data/tw_stocks.db` and hardcodes `/root/...` output paths — adjust those paths if running somewhere other than a freshly cloned `/root/Kronos` on a pod. Step 4 (backtest) is **not** parallelized this way yet — it's a much smaller workload (~100 signal dates vs. ~2477 extraction dates) so it was left sequential, but the same technique would apply if it ever becomes the bottleneck.

Original sequential form (kept for reference / smaller-scale runs where parallelizing isn't worth the complexity):

```bash
python -m finetune_tw.extract_embeddings --config finetune_tw/configs/config_tw_daily.yaml \
    --model pretrained --start 2015-01-01 --end 2023-12-31 --horizon 5 \
    --out finetune_tw/outputs/tw_daily/embeddings/train.parquet

python -m finetune_tw.extract_embeddings --config finetune_tw/configs/config_tw_daily.yaml \
    --model pretrained --start 2024-01-01 --end 2024-06-30 --horizon 5 \
    --out finetune_tw/outputs/tw_daily/embeddings/val.parquet
```

- [ ] **Step 3: Train XGBoost**

```bash
python -m finetune_tw.train_xgb_lambdarank \
    --train finetune_tw/outputs/tw_daily/embeddings/train.parquet \
    --val finetune_tw/outputs/tw_daily/embeddings/val.parquet \
    --out finetune_tw/outputs/tw_daily/xgb_round6.json
```

- [ ] **Step 4: Backtest against the same 2024-07-01+ test window as Round 0**

```bash
python -m finetune_tw.backtest_xgb_embedding \
    --config finetune_tw/configs/config_tw_daily.yaml \
    --model pretrained --xgb_model finetune_tw/outputs/tw_daily/xgb_round6.json \
    --hold_days_list 5 --top_k 10
```

- [ ] **Step 5: Compare against Round 0 baseline (Sharpe 1.12, Ann 38.59%, MaxDD 35.03%) and append a "Round 6" section to `docs/kronos-tw-round-history.md`**

Follow the exact structure of the "Round 5" section already in that file (起點 / 平台 / 驗證的方法 / 回測結果 table / 結論). State plainly whether Sharpe ≥ 1.12 (M1 beats Round 0) or not, and if not, note it alongside the other "已窮盡" entries so the next research iteration knows M1 was tried.

- [ ] **Step 6: Commit**

```bash
git add docs/kronos-tw-round-history.md autoresearch/tw-evals/finetune-tw-results.tsv
git commit -m "docs+feat: Round 6 results — Kronos Embedding + XGBoost LambdaRankIC"
```

---

## Self-Review

**1. Spec coverage:** M1 from `improvement-plan.md` requires (a) extract Kronos hidden states — Task 1; (b) train XGBoost with LambdaRankIC — Tasks 2+3; (c) backtest via existing `backtest_next_open.py`-style harness at top_k=10/hold=5d — Task 4; (d) run it and record results — Task 7. All plan bullets are covered. The plan's own "DECISION NEEDED" (mean pooling vs last-token) is resolved by defaulting to mean pooling per the plan's own recommendation ("建議先試 mean pooling"); layer/pooling choice is now an explicit, testable ablation knob (Task 6) rather than a hardcoded assumption. The other "DECISION NEEDED" bullet (raw features alongside hidden state) is implemented directly in Task 5 rather than deferred, since it's cheap and additive. `research-findings.md`'s Insight #8 (layer depth carries different structure) motivates Task 6. Insights #4/#5/#6/#7 and `summary.md`'s "M2" alternative are fine-tuning-based and out of scope for this frozen-Kronos plan — see the "Additions from autoresearch/improve-260701-1512/" section above for why.

**2. Placeholder scan:** No TBD/TODO; every step shows complete, runnable code; no "similar to Task N" shortcuts — Task 4's helper reuses Task 1's `extract_embeddings_batch` by import, not by paraphrase.

**3. Type consistency:** `extract_embeddings_batch(predictor, df_list, x_timestamp_list) -> np.ndarray (B, d_model)` is defined once in Task 1 and consumed identically (same 3 positional args) in Task 4's `compute_xgb_signals`; Task 6 adds a 4th parameter `layer_indices=None` with a default that preserves this exact call signature at every existing call site (Task 4's call is untouched). `lambdarank_ic_objective(group_sizes, sigma)` from Task 2 is consumed with the same signature in Task 3's `train()`. `rank_ic` from `ic_validation.py` keeps its existing signature throughout — no renaming. `_feature_columns` (Task 3) and `_TECH_FEATURE_COLUMNS` (introduced in Task 5) are consumed with the same names in Task 5's `backtest_xgb_embedding.py` update — training and inference build the feature vector in the same `emb_* + feat_*` column order in both places, which is required for the booster's `DMatrix` to line up with what it was trained on.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-02-kronos-embedding-xgb-lambdarank.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
