# Batch 3b Validation Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Batch 3b-style independent train/validation date filtering to the streaming XGBoost trainer so one enriched parquet can be split into a long train range and a long validation range without changing the existing full-universe top-k evaluation path.

**Architecture:** Keep the current streaming trainer intact and add a thin date-filter layer around the existing parquet readers. Replace the single `keep_dates` input with independent `train_keep_dates` / `val_keep_dates`, add CLI helpers for `--train-start/--train-end` and `--val-start/--val-end`, and persist the effective split metadata in the summary sidecar.

**Tech Stack:** Python, pandas, pyarrow parquet streaming, xgboost, pytest

---

### Task 1: Lock the Batch 3b split behavior with failing tests

**Files:**
- Modify: `tests/finetune_tw/test_train_xgb_streaming.py`
- Test: `tests/finetune_tw/test_train_xgb_streaming.py`

- [ ] **Step 1: Write the failing test for independent train/val filters**

```python
def test_train_streaming_uses_independent_train_and_val_date_filters(tmp_path):
    train_df = _make_synthetic_df(n_dates=6, n_symbols=20, seed=21)
    path = tmp_path / "all.parquet"
    train_df.to_parquet(path, index=False)

    _, summary = train_streaming(
        path,
        path,
        feature_set="emb",
        train_keep_dates={"2024-01-01", "2024-01-02", "2024-01-03"},
        val_keep_dates={"2024-01-05", "2024-01-06"},
        num_boost_round=20,
        early_stopping_rounds=5,
        n_threads=2,
    )

    assert summary["train_dates"] == 3
    assert summary["val_dates"] == 2
    assert summary["train_rows"] == 60
    assert summary["val_rows"] == 40
```

- [ ] **Step 2: Write the failing test for CLI date-range helper behavior**

```python
def test_resolve_date_filter_range_without_trading_calendar():
    keep_dates = resolve_date_filter(
        start="2024-01-02",
        end="2024-01-04",
        trading_days=None,
    )
    assert keep_dates == {"2024-01-02", "2024-01-03", "2024-01-04"}


def test_resolve_date_filter_intersects_trading_calendar():
    keep_dates = resolve_date_filter(
        start="2024-01-02",
        end="2024-01-05",
        trading_days={"2024-01-01", "2024-01-03", "2024-01-05"},
    )
    assert keep_dates == {"2024-01-03", "2024-01-05"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest -q tests/finetune_tw/test_train_xgb_streaming.py -k "independent_train_and_val_date_filters or resolve_date_filter"`

Expected: FAIL because `train_streaming` does not yet accept `train_keep_dates` / `val_keep_dates`, and `resolve_date_filter` does not exist.

---

### Task 2: Implement independent streaming filters and Batch 3b CLI surface

**Files:**
- Modify: `finetune_tw/train_xgb_streaming.py`
- Modify: `tests/finetune_tw/test_train_xgb_streaming.py`
- Test: `tests/finetune_tw/test_train_xgb_streaming.py`

- [ ] **Step 1: Add a reusable date-range resolver**

```python
def resolve_date_filter(
    start: str | None,
    end: str | None,
    trading_days: set[str] | None,
) -> set[str] | None:
    if start is None and end is None:
        return trading_days
    if start is None or end is None:
        raise ValueError("date filter requires both start and end")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("date filter start must be <= end")
    if trading_days is not None:
        return {d for d in trading_days if start <= d <= end}
    days = pd.date_range(start, end, freq="D")
    return {day.strftime("%Y-%m-%d") for day in days}
```

- [ ] **Step 2: Split trainer inputs into train and validation filters**

```python
def train_streaming(
    train_path,
    val_path,
    feature_set: str,
    train_keep_dates: set[str] | None,
    val_keep_dates: set[str] | None,
    ...
):
    train_groups = scan_group_sizes(train_path, train_keep_dates)
    dtrain = xgb.QuantileDMatrix(
        ParquetIter(train_path, feat_cols, train_keep_dates),
        ...
    )
    dval, val_groups = load_val_matrix(val_path, feat_cols, val_keep_dates)
```

- [ ] **Step 3: Add Batch 3b CLI args and summary metadata**

```python
parser.add_argument("--train-start")
parser.add_argument("--train-end")
parser.add_argument("--val-start")
parser.add_argument("--val-end")

trading_days = None if args.no_twse_filter else twse_trading_days(args.db)
train_keep_dates = resolve_date_filter(args.train_start, args.train_end, trading_days)
val_keep_dates = resolve_date_filter(args.val_start, args.val_end, trading_days)
```

Also record the effective split metadata in the summary:

```python
"train_start": args.train_start,
"train_end": args.train_end,
"val_start": args.val_start,
"val_end": args.val_end,
```

- [ ] **Step 4: Run targeted tests to verify the new behavior**

Run: `pytest -q tests/finetune_tw/test_train_xgb_streaming.py`

Expected: PASS

- [ ] **Step 5: Run focused CLI-free regression verification**

Run: `pytest -q tests/finetune_tw/test_train_xgb_streaming.py -k "improves_rank_ic or accepts_multiple_validation_parquets"`

Expected: PASS
