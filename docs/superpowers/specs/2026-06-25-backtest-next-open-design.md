# Next-Open Backtest Design

**Date:** 2026-06-25
**Status:** Approved

## Overview

Add a second backtest entry point that keeps the existing Kronos signal-generation logic, CLI shape, JSON schema, and chart layout, but aligns portfolio execution with intended live usage:

- Signal is produced after the close on trading day `T`
- Orders are assumed to execute at the open on the next trading day `T+1`
- Transaction costs, tax, and slippage are not modeled

The current `finetune_tw/backtest.py` remains unchanged. The new behavior lives in a separate module so both assumptions remain available and explicit.

## Requirements

| Dimension | Decision |
|-----------|----------|
| Existing backtest | Keep unchanged |
| New module | `finetune_tw/backtest_next_open.py` |
| Signal logic | Same as `signal_today.py` / current `backtest.py` |
| Trading calendar | Real exchange dates from DB benchmark symbol, not `pd.bdate_range` |
| Entry timing | `T+1` open |
| Exit / rebalance timing | Next rebalance day open |
| Costs | None |
| Output JSON schema | Same structure as current backtest |
| Output filenames | Distinct suffix to avoid overwrite |

## Architecture

### Separate entry point

Create `finetune_tw/backtest_next_open.py` as a sibling of the existing `backtest.py`.

Why separate:

- avoids mixing two execution assumptions behind one flag
- keeps regression risk low for existing workflows
- makes result provenance obvious from module name and output filenames

### Shared behavior to preserve

The new module must preserve these behaviors exactly:

- same model loading path and model choices
- same context window construction from DB rows
- same six input features: `open`, `high`, `low`, `close`, `volume`, `amount`
- same predictor inference parameters: `T=1.0`, `top_k=1`, `top_p=1.0`, `sample_count=1`
- same top-k ranking and threshold filtering logic
- same chart layout and metrics computation shape

This keeps signal generation aligned with `signal_today.py`. The intended change is execution timing, not forecasting logic.

## Trading Calendar

### Current issue

The existing backtest uses `pd.bdate_range(...)`, which can include business days that are not Taiwan exchange trading days. That can misalign rebalance anchors and hold windows relative to both the DB and live operation.

### New rule

The next-open backtest must derive its calendar from the benchmark symbol stored in SQLite, normally `^TWII`.

Implementation requirements:

- load benchmark rows from DB over the backtest date range
- use the benchmark `date` column as the authoritative trading calendar
- generate rebalance dates by subsampling this actual trading-date index
- use the same calendar to determine each rebalance day’s next tradable open

If a rebalance date does not have a following trading day, it is not tradable and must be excluded from holdings/performance construction.

## Signal Generation

### Reuse existing inference semantics

For each rebalance date `T`, the new module must compute signals the same way as the current backtest:

1. load each symbol’s recent history up to and including `T`
2. require at least `cfg.lookback_window` rows
3. build the same predictor inputs
4. predict the future close path
5. define signal as predicted `close(T + hold_days)` divided by current `close(T)`, minus 1

This means a stock selected on day `T` is selected using only information available after the `T` close.

### Scope of reuse

The implementation should reuse existing helper functions where practical, but correctness takes priority over deduplication. Small wrappers or copied helper logic are acceptable if they reduce coupling to the old close-to-close execution code.

## Execution Logic

### Portfolio timing

For a rebalance signal generated on trading day `T`:

- holdings become active at the next trading day open, `T+1`
- holdings stay active until the next rebalance’s execution open

`hold_days` is defined as the number of full trading sessions the portfolio remains invested after its execution open.

That means:

- signal anchor `T` is the close that decides the next holdings
- execution day `E` is the next trading day after `T`
- for `hold_days = h`, the portfolio fully owns the sessions `E, E+1, ..., E+h-1`
- the next rebalance signal is generated after the close of the last fully held session
- the next rebalance execution happens at the following trading day open

Equivalently, rebalance signal anchors remain spaced every `h` trading days on the benchmark calendar, but realized returns begin one trading day later at the corresponding execution open.

### Return construction

The portfolio return engine must use both open and close series.

Return construction is defined at the portfolio level around each execution open.

For one rebalance cycle:

1. Signal is observed on anchor day `T`
2. Execution day `E` is the next trading day after `T`
3. The next rebalance anchor `T_next` is the close of the last fully held session
4. The next execution day `E_next` is the next trading day after `T_next`

Portfolio daily returns are then built as follows:

1. On execution day `E`:
   - portfolio intraday return is computed from the newly selected holdings using `close(E) / open(E) - 1`
2. For each trading day `D` from the first held session after entry through `T_next` close:
   - portfolio daily return is computed from the active holdings using `close(D) / close(prev_trading_day) - 1`
3. On the next execution day `E_next`:
   - the overnight gap from the previous holdings is realized first as `open(E_next) / close(prev_trading_day) - 1`
   - after rebalancing at that open, the same day’s intraday return belongs to the new holdings, not the old holdings

This avoids ambiguity when a symbol remains in the portfolio across consecutive rebalances. A continuing symbol may appear in both the old and new holdings, but the portfolio is still treated as rebalanced at `E_next` open.

Equal-weight portfolio return remains the simple mean of constituent returns for the applicable holdings on each segment.

For internal helper outputs that report one return per rebalance interval, the interval return should represent the outgoing holdings only, measured from the current execution open to the next execution open.

### Interpretation

This models a portfolio that:

- observes the signal after close on `T`
- enters at `T+1` open
- remains invested for exactly `hold_days` full trading sessions after entry
- marks to market on closes while the position is open
- realizes the overnight gap into the next rebalance open using the outgoing holdings
- applies the rebalance day intraday move using the incoming holdings

It does not attempt intraday execution beyond open/close prices.

## Output Contract

### CLI

The new module should mirror the existing CLI shape:

```bash
python -m finetune_tw.backtest_next_open \
  --config finetune_tw/configs/config_tw_daily.yaml \
  --model round2 \
  --hold_days_list 5 10 15
```

Supported flags remain:

- `--config`
- `--model`
- `--hold_days_list`
- `--top_k`
- `--test_start`
- `--threshold`

### Output files

To avoid overwriting existing backtest artifacts:

- JSON: `backtest_returns_{model_key}_next_open.json`
- PNG: `backtest_{model_key}_next_open.png`

### JSON schema

Keep the same top-level structure and nested metric fields as the current backtest output:

- `model_key`
- `model_label`
- `test_start`
- `test_end`
- `top_k`
- `hold_variants`
- `benchmark`

This allows existing consumers to continue working with minimal or no changes.

The only semantic difference is that `hold_variants[*].daily_returns` now represent next-open execution returns instead of close-to-close returns.

Benchmark daily returns remain benchmark close-to-close returns on the actual benchmark trading calendar.

## Error Handling

- Symbols missing sufficient context rows are skipped, same as current behavior
- Symbols missing required open/close rows during a holding interval are excluded from that interval’s contribution
- Rebalance anchors without a following trading day are dropped
- If no valid portfolio daily returns are produced for a hold variant, metrics should still return a valid empty-series-safe result or fail with a clear error rather than silently writing misleading output

## Testing

Add targeted tests in `tests/finetune_tw/test_backtest_next_open.py`.

Required coverage:

1. Trading calendar comes from actual benchmark dates, not business-day synthesis
2. Signal date `T` maps to execution date `T+1`
3. Daily return construction uses `entry open -> same-day close`, then `close -> close`, then `prev close -> next rebalance open`, then rebalance-day `open -> close` for the new holdings
4. Equal-weight portfolio aggregation matches expected arithmetic
5. Output schema matches the current backtest structure
6. Output filenames use the `_next_open` suffix
7. Final rebalance anchor without a next trading day is excluded

Tests should use small synthetic price tables so entry/exit math is explicit and reviewable.

## Out of Scope

- transaction costs
- tax
- slippage
- volume participation limits
- limit-up / limit-down execution constraints
- partial fills
- refactoring the old and new backtests into a shared abstraction layer

## Implementation Notes

Prefer the smallest change that preserves clarity:

- keep existing `backtest.py` behavior frozen
- build only the minimal new helpers needed for next-open execution
- reuse plotting and metrics code if convenient
- do not change `signal_today.py` as part of this work

## Success Criteria

The work is successful when:

1. `python -m finetune_tw.backtest_next_open ...` runs independently of the existing backtest
2. the selected holdings for any rebalance date match the current signal-generation logic
3. realized returns reflect `T close -> T+1 open` execution timing
4. artifacts are saved without overwriting existing backtest outputs
5. tests cover the execution-timing behavior and pass
