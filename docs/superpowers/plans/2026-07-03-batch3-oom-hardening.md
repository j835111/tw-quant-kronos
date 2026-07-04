# Batch 3 OOM Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Batch 3 feature enrichment, retraining, diagnostics, and inference paths stay comfortably within the 13GB host RAM budget, with explicit safety margin instead of “usually works”.

**Architecture:** Keep the existing Batch 2 streaming training shape, but remove new whole-dataset materializations introduced by Batch 3. The core strategy is to process data in date-complete streaming blocks, preallocate once when unavoidable, and never rebuild or rescan full lookup tables inside inner loops.

**Tech Stack:** Python 3.12, pandas, pyarrow, xgboost, pytest

---

## File Map

- Modify: `finetune_tw/enrich_round6_features.py`
  - Replace whole-artifact feature lookup with date-block streaming enrichment.
- Modify: `finetune_tw/feature_engineering.py`
  - Add helpers that compute features / cs-ranks for bounded blocks without forcing whole-table copies.
- Modify: `finetune_tw/train_xgb_streaming.py`
  - Make validation loading and group-size scanning single-pass and bounded.
- Modify: `finetune_tw/round6_diagnostics.py`
  - Add parquet column projection and date-stream aggregation so diagnostics no longer hold all scored rows in memory.
- Modify: `finetune_tw/backtest_xgb_embedding.py`
  - Restore fast path for models that do not consume `*_cs_rank`, so inference does not pay unnecessary per-date ranking costs.
- Optional follow-up modify: `finetune_tw/extract_embeddings.py`
  - Avoid end-of-job full-frame `add_cross_sectional_rank_features()` if this path will be used again for Batch 4.
- Create/Modify tests:
  - `tests/finetune_tw/test_train_xgb_streaming.py`
  - `tests/finetune_tw/test_round6_diagnostics.py`
  - `tests/finetune_tw/test_feature_engineering.py`
  - `tests/finetune_tw/test_extract_embeddings.py` only if the extraction path is included in this pass.

---

### Task 1: Make Parquet Enrichment Truly Batch-Bounded

**Files:**
- Modify: `finetune_tw/enrich_round6_features.py`
- Modify: `finetune_tw/feature_engineering.py`
- Test: `tests/finetune_tw/test_feature_engineering.py`

- [ ] **Step 1: Write failing tests for date-complete block enrichment helpers**

Add tests that lock in two behaviors:
1. Enrichment works on a bounded block of rows without requiring global `(date, symbol)` tables.
2. `cs_rank` is computed correctly when a date is split across record batches but reassembled before ranking.

Example tests to add:

```python
def test_add_cross_sectional_rank_features_requires_complete_date_block():
    df = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        "symbol": ["AAA", "BBB", "AAA", "BBB"],
        "feat_ma5_dist": [0.1, 0.3, 0.9, 0.2],
    })
    ranked = add_cross_sectional_rank_features(df, feature_cols=["feat_ma5_dist"])
    assert ranked["feat_ma5_dist_cs_rank"].tolist() == pytest.approx([0.5, 1.0, 1.0, 0.5])


def test_enrichment_helper_merges_recomputed_features_for_one_date_block():
    block = pd.DataFrame({
        "date": ["2024-03-20", "2024-03-20"],
        "symbol": ["AAA", "BBB"],
        "label": [0.1, 0.2],
        "emb_0": [1.0, 2.0],
    })
    feature_block = pd.DataFrame({
        "date": ["2024-03-20", "2024-03-20"],
        "symbol": ["AAA", "BBB"],
        "feat_ma5_dist": [0.01, 0.02],
        "feat_ma5_dist_cs_rank": [0.5, 1.0],
    })
    merged = _merge_feature_block(block, feature_block)
    assert "feat_ma5_dist" in merged.columns
    assert merged["feat_ma5_dist_cs_rank"].tolist() == [0.5, 1.0]
```

- [ ] **Step 2: Run tests to confirm they fail for the current whole-table design**

Run:

```bash
pytest -q tests/finetune_tw/test_feature_engineering.py
```

Expected: failure because `_merge_feature_block` / date-block helpers do not exist yet.

- [ ] **Step 3: Refactor enrichment to operate on date-complete streaming blocks**

Implement this structure in `finetune_tw/enrich_round6_features.py`:

```python
def _iter_date_blocks(parquet_path: str, keep_dates: set[str] | None, batch_size: int):
    pf = pq.ParquetFile(parquet_path)
    pending = []
    current_date = None
    for batch in pf.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        chunk["date"] = _date_strings(chunk["date"])
        if keep_dates is not None:
            chunk = chunk.loc[chunk["date"].isin(keep_dates)].copy()
        if chunk.empty:
            continue
        for date, g in chunk.groupby("date", sort=False):
            if current_date is None:
                current_date = date
            if date != current_date:
                yield current_date, pd.concat(pending, ignore_index=True)
                pending = [g]
                current_date = date
            else:
                pending.append(g)
    if pending:
        yield current_date, pd.concat(pending, ignore_index=True)


def _feature_block_for_dates(db_path: str, block: pd.DataFrame, buffer_days: int) -> pd.DataFrame:
    symbols = sorted(block["symbol"].unique())
    min_date = pd.Timestamp(block["date"].min()) - pd.Timedelta(days=buffer_days)
    max_date = pd.Timestamp(block["date"].max())
    history = query_symbols_window(
        db_path,
        symbols,
        start=min_date.strftime("%Y-%m-%d"),
        end=max_date.strftime("%Y-%m-%d"),
    )
    feature_df = compute_technical_feature_frame(history)
    feature_df = feature_df.merge(
        block[["date", "symbol"]].drop_duplicates(),
        on=["date", "symbol"],
        how="inner",
        validate="1:1",
    )
    return add_cross_sectional_rank_features(feature_df, feature_cols=TECH_FEATURE_COLUMNS)
```

Rules for the implementation:
- Never build a whole-artifact `keys` DataFrame.
- Never build a whole-artifact `feature_lookup`.
- Never call `reset_index()` on a giant lookup table inside the write loop.
- Peak working set must be bounded by one date block plus that block’s trailing history window.

- [ ] **Step 4: Verify the feature-engineering tests pass**

Run:

```bash
pytest -q tests/finetune_tw/test_feature_engineering.py
```

Expected: PASS

- [ ] **Step 5: Smoke-test enrichment on a tiny parquet fixture**

Run:

```bash
python3 -m finetune_tw.enrich_round6_features \
  --input /tmp/in.parquet \
  --output /tmp/out.parquet \
  --db /tmp/tw.db \
  --batch-size 2 \
  --no-twse-filter
```

Expected: command succeeds and `/tmp/out.parquet` contains recomputed `feat_*` and `*_cs_rank` columns.

---

### Task 2: Remove Avoidable Dense Copies from Streaming Training

**Files:**
- Modify: `finetune_tw/train_xgb_streaming.py`
- Test: `tests/finetune_tw/test_train_xgb_streaming.py`

- [ ] **Step 1: Write failing tests for single-pass validation loading and streamed group counting**

Add coverage for:
1. multi-validation input still returns the right row/date totals;
2. group-size scanning works from streamed date batches, not full-column pandas materialization.

Example tests:

```python
def test_scan_group_sizes_streams_date_batches(tmp_path):
    df = _make_synthetic_df(n_dates=4, n_symbols=10)
    path = tmp_path / "t.parquet"
    df.to_parquet(path, index=False)
    assert scan_group_sizes(path, None) == [10, 10, 10, 10]


def test_train_streaming_accepts_multiple_validation_parquets(tmp_path):
    ...
    _, summary = train_streaming(train_path, [val_a_path, val_b_path], feature_set="emb", keep_dates=None)
    assert summary["val_rows"] == len(val_a) + len(val_b)
    assert summary["val_dates"] == 5
```

- [ ] **Step 2: Run the training tests and capture the current failure or inefficiency target**

Run:

```bash
pytest -q tests/finetune_tw/test_train_xgb_streaming.py
```

Expected: either failure in new tests or current implementation still depending on dense re-materialization.

- [ ] **Step 3: Rework validation loading and group-size scanning to be one-pass**

Implement:

```python
def scan_group_sizes(parquet_path, keep_dates):
    pf = pq.ParquetFile(parquet_path)
    sizes = []
    prev_date = None
    run_len = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["date"]):
        dates = _date_strings(batch.column(0).to_pandas())
        if keep_dates is not None:
            dates = dates[dates.isin(keep_dates)]
        for date in dates:
            if prev_date is None or date == prev_date:
                run_len += 1
            else:
                sizes.append(run_len)
                run_len = 1
            prev_date = date
    if run_len:
        sizes.append(run_len)
    return sizes


def load_val_matrix(parquet_path, feat_cols, keep_dates):
    if isinstance(parquet_path, (list, tuple)):
        group_lists = [scan_group_sizes(path, keep_dates) for path in parquet_path]
        total_rows = sum(sum(groups) for groups in group_lists)
        x = np.empty((total_rows, len(feat_cols)), dtype=np.float32)
        y = np.empty(total_rows, dtype=np.float32)
        pos = 0
        all_groups = []
        for path, groups in zip(parquet_path, group_lists):
            pos = _fill_matrix_from_parquet(path, feat_cols, keep_dates, x, y, pos)
            all_groups.extend(groups)
        dval = xgb.DMatrix(x, label=y)
        dval.set_group(all_groups)
        return dval, all_groups
```

Rules:
- Do not call `get_data().toarray()`.
- Do not build intermediate `DMatrix` shards only to unpack them again.
- Keep the previous safety property: one dense validation matrix total, not several copies.

- [ ] **Step 4: Verify training tests pass**

Run:

```bash
pytest -q tests/finetune_tw/test_train_xgb_streaming.py
```

Expected: PASS

---

### Task 3: Make Diagnostics Column-Projected and Date-Streamed

**Files:**
- Modify: `finetune_tw/round6_diagnostics.py`
- Test: `tests/finetune_tw/test_round6_diagnostics.py`

- [ ] **Step 1: Write failing tests for raw-only projection and bounded daily aggregation**

Add tests that lock in:
1. raw-only runs request only `date/symbol/label/feat_*` plus the selected score columns;
2. daily metrics can be accumulated date-by-date without concatenating the full scored frame first.

Example tests:

```python
def test_feature_set_columns_raw_excludes_embeddings():
    cols = ["date", "symbol", "label", "emb_0", "feat_ma5_dist", "feat_ma5_dist_cs_rank"]
    assert feature_set_columns(cols, "raw") == ["feat_ma5_dist", "feat_ma5_dist_cs_rank"]


def test_per_day_metrics_handles_one_date_frame():
    df = _scored_frame(n_symbols=30, dates=("2024-01-02",))
    daily = per_day_metrics(df, score_col="score", top_k=5)
    assert len(daily) == 1
```

- [ ] **Step 2: Run diagnostics tests to confirm the new behavior is not implemented yet**

Run:

```bash
pytest -q tests/finetune_tw/test_round6_diagnostics.py
```

Expected: failure in the new projection/streaming-oriented tests.

- [ ] **Step 3: Refactor `stream_scores()` into a projected, date-streaming pipeline**

Implement the following shape:

```python
def stream_scores(...):
    projected = ["date", "symbol", "label", *technical_feature_columns(pf.schema_arrow.names), *feat_cols]
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(dict.fromkeys(projected))):
        ...


def iter_scored_dates(...):
    pending = []
    current_date = None
    for scored_chunk in stream_scored_batches(...):
        for date, g in scored_chunk.groupby("date", sort=False):
            ...
            yield completed_date_frame
```

Rules:
- Raw-only diagnostics must not decode `emb_*`.
- Write scored parquet incrementally with `ParquetWriter` instead of keeping `outs` in memory.
- Compute daily metrics once per completed date frame.
- Build quarterly/monthly summaries from compact daily-metrics tables, not from the full scored dataset.

- [ ] **Step 4: Verify diagnostics tests pass**

Run:

```bash
pytest -q tests/finetune_tw/test_round6_diagnostics.py
```

Expected: PASS

---

### Task 4: Restore Inference Fast Path and Fence Off Non-Critical Extraction Risk

**Files:**
- Modify: `finetune_tw/backtest_xgb_embedding.py`
- Optional modify: `finetune_tw/extract_embeddings.py`
- Test: `tests/finetune_tw/test_extract_embeddings.py` only if extraction is changed

- [ ] **Step 1: Add a fast path for models that do not use `*_cs_rank`**

Update `_assemble_feature_matrix()` / `compute_xgb_signals()` so that:
- if `feature_columns` is absent or contains no `*_cs_rank`, predict batch-by-batch exactly like the old path;
- only models that explicitly require rank features pay the per-date materialization cost.

Sketch:

```python
needs_cs_rank = feature_columns is not None and any(c.endswith("_cs_rank") for c in feature_columns)
if not needs_cs_rank:
    tech_feats = np.array([[compute_technical_features(df)[c] for c in _TECH_FEATURE_COLUMNS] for df in sub_dfs], dtype=np.float32)
    features = np.concatenate([embeddings, tech_feats], axis=1)
    preds = booster.predict(xgb.DMatrix(features))
else:
    ...  # current per-date rank-aware path
```

- [ ] **Step 2: Decide whether to harden `extract_embeddings.py` in this pass**

If this path will be reused before Batch 4, refactor it to emit date-complete blocks and rank them before append; otherwise leave the code unchanged but add a comment documenting that Batch 3 production flow must use `enrich_round6_features.py`, not `extract_embeddings.py`, to avoid a whole-frame late-stage `cs_rank` pass.

- [ ] **Step 3: Verify no regressions in extraction/inference tests**

Run:

```bash
pytest -q tests/finetune_tw/test_extract_embeddings.py
```

Expected: PASS

---

### Task 5: End-to-End Safety Verification on Realistic Inputs

**Files:**
- Modify only if any gaps are found during verification

- [ ] **Step 1: Re-run the targeted unit suite**

Run:

```bash
pytest -q \
  tests/finetune_tw/test_feature_engineering.py \
  tests/finetune_tw/test_train_xgb_streaming.py \
  tests/finetune_tw/test_round6_diagnostics.py \
  tests/finetune_tw/test_extract_embeddings.py
```

Expected: all pass

- [ ] **Step 2: Smoke-test bounded enrichment on the ext4 artifact copies**

Run on a small slice first:

```bash
python3 -m finetune_tw.enrich_round6_features \
  --input /home/james/round6_artifacts/embeddings_val_rechunked.parquet \
  --output /home/james/round6_artifacts/embeddings_val_batch3.parquet \
  --db /home/james/round6_artifacts/tw_stocks.db \
  --batch-size 50000
```

Expected:
- command completes without rapid RSS climb;
- output parquet exists and includes `feat_*` plus `*_cs_rank`.

- [ ] **Step 3: Smoke-test training with top-tail-aware selection metric**

Run:

```bash
python3 -m finetune_tw.train_xgb_streaming \
  --features full \
  --train /home/james/round6_artifacts/embeddings_train_rechunked_batch3.parquet \
  --val /home/james/round6_artifacts/embeddings_val_batch3.parquet \
  --db /home/james/round6_artifacts/tw_stocks.db \
  --selection-metric top_k_excess \
  --top-k 10 \
  --mem-limit-gb 12 \
  --out /tmp/xgb_batch3_smoke.json
```

Expected:
- RSS remains well below the 12GB cap with visible headroom;
- process exits 0;
- summary json records `feature_columns`, `selection_metric`, and top-tail metrics.

- [ ] **Step 4: Only after the smoke passes, queue the full Batch 3 run**

Required precondition:
- enrichment is date-streamed and stable;
- training still respects the 12GB RLIMIT ceiling;
- diagnostics no longer decode `emb_*` for raw-only ablations.

---

## Spec Coverage Check

- OOM safety on enrichment: covered by Task 1.
- OOM safety on training/validation: covered by Task 2.
- Diagnostics memory / raw-only waste: covered by Task 3.
- Inference regression and fallback path: covered by Task 4.
- Safety margin validation before full run: covered by Task 5.

## Placeholder Scan

- No `TODO` / `TBD` markers.
- Each task lists exact files and concrete commands.
- Each risky path from the subagent review has a corresponding task.

## Type / Interface Consistency

- `technical_feature_columns()` remains the single source of truth for `feat_*` ordering.
- `train_streaming(..., val_path)` must continue accepting a single parquet path and now also a list/tuple of paths.
- `feature_columns` sidecar remains the contract between training and inference for any model that consumes `*_cs_rank`.
