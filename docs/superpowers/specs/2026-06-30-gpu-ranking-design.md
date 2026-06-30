# GPU-Centric Ranking/Oracle Refactor Design

## Goal

Reduce CPU work introduced after `49cc85c7208e99f726582c77ff2012e434e22e1f`, keep most compute on GPU during training, and preserve current function signatures plus config compatibility.

## Scope

This design covers three hot paths:

1. `CrossSectionalDateSampler` used by auxiliary ranking loss
2. S1 oracle construction used at training startup
3. Gradient accumulation behavior in `run_training`

Public function signatures, training entry points, and YAML/config fields remain unchanged.

## Current Problems

### Ranking sampler

`CrossSectionalDateSampler.sample_date_batch()` currently performs repeated SQLite queries, pandas conversions, sorting, and per-symbol normalization inside the training step. It also samples from `pd.bdate_range(...)`, which includes non-trading weekdays and can produce empty ranking batches after scanning the universe.

### Oracle build

Oracle construction currently normalizes windows on CPU, tokenizes in batches, then falls back to Python-level sample iteration and CPU-side accumulation. This adds substantial host overhead during startup and leaves vectorizable aggregation unused.

### Gradient accumulation

`grad_accum_steps` is read from config but not applied as true accumulation because each iteration still clears gradients and steps the optimizer immediately.

## Recommended Approach

Use a mixed in-memory and batched-tensor refactor:

1. Move cross-sectional ranking data access from per-step DB queries to one-time in-memory preload.
2. Replace oracle sample-by-sample accumulation with batched tensor accumulation.
3. Implement real optimizer stepping across `grad_accum_steps` microbatches.

This removes the dominant CPU bottlenecks without changing call sites or adding new cache invalidation complexity.

## Detailed Design

### 1. Cross-sectional sampler internals

`CrossSectionalDateSampler` keeps its constructor and `sample_date_batch()` interface unchanged.

Internal changes:

- Load symbol data once during initialization.
- Store per-symbol feature arrays as `float32` numpy arrays.
- Store per-symbol date arrays in sorted order and precomputed stamp arrays.
- Build the sampler date pool from actual dates present in loaded data, filtered to dates that can satisfy the requested lookback and horizon.
- Maintain a lightweight mapping from date string to candidate symbol/index pairs to avoid rescanning the full universe during each step.

Sampling flow:

- Draw one valid trading date from the precomputed date pool.
- Select up to `n_stocks` valid symbol/index pairs for that date.
- Slice context/future windows directly from numpy arrays.
- Perform normalization on array slices without pandas.
- Return the same dictionary keys and tensor shapes as today.

Expected result:

- No SQLite or pandas work in the ranking loss training step.
- No wasted scans on holidays or other non-trading weekdays.

### 2. Oracle construction internals

`_iter_s1_oracle_samples()` and `build_s1_oracle_from_samples()` keep their current signatures.

Internal changes:

- `_iter_s1_oracle_samples()` will continue to emit the current sample structure for compatibility, but it will avoid unnecessary copies where possible.
- `build_s1_oracle_from_samples()` will detect dictionary samples with `s1_ids` and `open_prices`, batch them, and compute:
  - last token ids
  - realized returns
  - token counts
  - token return sums
  using tensor operations instead of Python accumulation.
- Use `scatter_add_` or equivalent indexed accumulation on tensors.
- Keep fallback handling for edge cases and unsupported sample shapes so existing tests and callers remain valid.

Expected result:

- Startup still does some host preprocessing, but the expensive aggregation becomes batched and vectorized.
- Fewer `.cpu()` round-trips and less Python overhead.

### 3. Gradient accumulation

`run_training()` will keep the current config field and outer loop structure.

Behavior changes:

- Scale token loss by `grad_accum_steps` as today.
- Accumulate gradients across multiple microbatches before stepping the optimizer.
- Run `optimizer.zero_grad(set_to_none=True)` only at accumulation boundaries.
- Step scheduler only when the optimizer steps.
- Handle the final partial accumulation window safely.

Ranking loss behavior:

- Ranking loss remains optional and controlled by existing config.
- The ranking step frequency continues to use the current config field.
- Ranking loss participates in the same backward pass as token loss for the current microbatch.

## Compatibility

The following remain unchanged:

- `CrossSectionalDateSampler.__init__`
- `CrossSectionalDateSampler.sample_date_batch`
- `_iter_s1_oracle_samples`
- `build_s1_oracle`
- `build_s1_oracle_from_samples`
- `run_training` signature
- Existing config/YAML fields
- Returned batch keys and tensor dtypes expected by tests

## Behavioral Parity

The implementation target is behavioral parity for existing public outputs and existing test expectations, with two explicit exceptions that are treated as bug fixes rather than compatibility breaks:

1. `grad_accum_steps` will begin working as configured, instead of stepping the optimizer every iteration.
2. Cross-sectional date sampling will use real tradable dates instead of calendar business days, so seeded sampling may choose a different valid date if a previous seed would have landed on a non-trading weekday.

Outside of those two corrections, the refactor is intended to preserve:

- function signatures
- config compatibility
- output dictionary structure
- tensor shapes and dtypes
- oracle numeric behavior for the same effective samples

## Testing Plan

Add or update tests to verify:

1. Cross-sectional sampling uses only valid trading dates and does not fail on non-trading weekdays in the calendar range.
2. Cross-sectional batches still return the same shapes, dtypes, and finite returns.
3. Oracle batching returns the same values as the prior sample-wise logic on representative fixtures.
4. Oracle paths remain compatible with current dict-based sample inputs.
5. Gradient accumulation performs optimizer stepping at the correct cadence.
6. Existing ranking integration tests still pass with the refactored sampler/oracle path.

## Risks

### Memory usage

Preloading cross-sectional data increases RAM usage. This is acceptable because the training dataset is already materially memory-backed, and the refactor trades RAM for much lower per-step CPU overhead.

### Date index correctness

The valid-date filtering must respect both lookback and horizon windows. Tests should cover boundary dates to prevent off-by-one errors.

### Accumulation semantics

Changing optimizer-step cadence can alter training behavior if the implementation mis-handles the last microbatch or scheduler stepping. Dedicated tests are required.

## Non-Goals

- No changes to public config names or training CLI usage
- No new disk cache format
- No predictor/tokenizer API redesign
- No broad refactor outside ranking/oracle/accumulation paths
