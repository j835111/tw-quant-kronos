# Next-Open Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate `finetune_tw.backtest_next_open` module that keeps current Kronos signal generation but evaluates portfolio returns with `T` close signal generation and `T+1` open execution on real benchmark trading dates.

**Architecture:** Keep the existing `finetune_tw/backtest.py` unchanged and implement the next-open variant in a new sibling module. Reuse current model-loading and signal-generation helpers from `finetune_tw.backtest`, but own the trading-calendar, next-open portfolio-return engine, output filenames, and CLI entry point inside the new module. `hold_days` means the number of full trading sessions owned after the entry open, so the next signal anchor is the close of the last fully held session and the next execution happens on the following trading day open.

**Tech Stack:** Python, pandas, NumPy, PyTorch, matplotlib, pytest, SQLite

---

## File Structure

**Create:**

- `finetune_tw/backtest_next_open.py`
  - Independent CLI entry point
  - Imports existing model/signal helpers from `finetune_tw.backtest`
  - Adds benchmark-calendar helpers
  - Adds next-open execution return engine
  - Saves `_next_open` JSON and PNG artifacts

- `tests/finetune_tw/test_backtest_next_open.py`
  - Synthetic DB fixtures and focused unit/integration tests
  - Covers real-trading-calendar behavior, `T+1` execution mapping, next-open return math, output schema, and suffixes

**Do not modify:**

- `finetune_tw/backtest.py`
- `finetune_tw/signal_today.py`

This keeps the old close-to-close backtest behavior frozen while adding the new execution assumption as a clearly separate tool.

### Task 1: Scaffold Trading Calendar Helpers

**Files:**
- Create: `tests/finetune_tw/test_backtest_next_open.py`
- Create: `finetune_tw/backtest_next_open.py`

- [ ] **Step 1: Write the failing tests for trading-calendar loading and `T+1` execution mapping**

```python
from __future__ import annotations

import pandas as pd
import pytest

from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices


def _seed_calendar_db(tmp_path) -> str:
    db_path = str(tmp_path / "calendar.db")
    init_db(db_path)

    benchmark = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1_000.0] * 5,
            "amount": [100_500.0, 101_500.0, 102_500.0, 103_500.0, 104_500.0],
        }
    )
    upsert_prices(db_path, "^TWII", benchmark)
    return db_path


def test_load_trading_calendar_uses_benchmark_dates(tmp_path):
    import finetune_tw.backtest_next_open as bo

    db_path = _seed_calendar_db(tmp_path)
    cfg = Config(
        db_path=db_path,
        benchmark_symbol="^TWII",
        test_start_date="2024-01-01",
    )

    dates = bo._load_trading_calendar(cfg, end="2024-01-31")

    assert list(dates.strftime("%Y-%m-%d")) == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    ]


def test_build_signal_and_execution_dates_drops_last_anchor_without_next_day():
    import finetune_tw.backtest_next_open as bo

    trading_dates = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"]
    )

    signal_dates, execution_dates = bo._build_signal_and_execution_dates(
        trading_dates,
        hold_days=2,
    )

    assert list(signal_dates.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-05"]
    assert list(execution_dates.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-08"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/finetune_tw/test_backtest_next_open.py::test_load_trading_calendar_uses_benchmark_dates tests/finetune_tw/test_backtest_next_open.py::test_build_signal_and_execution_dates_drops_last_anchor_without_next_day -v`

Expected: `FAIL` with `ModuleNotFoundError: No module named 'finetune_tw.backtest_next_open'`

- [ ] **Step 3: Write the minimal module scaffold and calendar helpers**

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch

from finetune_tw.backtest import (
    build_model_specs,
    compute_metrics,
    compute_raw_signals,
    load_predictor_from_spec,
    rank_stocks,
    signals_to_holdings,
)
from finetune_tw.config import Config
from finetune_tw.db import list_symbols, query_symbol


def _load_trading_calendar(cfg: Config, end: str) -> pd.DatetimeIndex:
    bm_df = query_symbol(
        cfg.db_path,
        cfg.benchmark_symbol,
        start=cfg.test_start_date,
        end=end,
    )
    if bm_df.empty:
        raise ValueError(
            f"No benchmark rows found for {cfg.benchmark_symbol} between "
            f"{cfg.test_start_date} and {end}."
        )
    return pd.DatetimeIndex(pd.to_datetime(bm_df["date"]))


def _build_signal_and_execution_dates(
    trading_dates: pd.DatetimeIndex,
    hold_days: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if hold_days <= 0:
        raise ValueError(f"hold_days must be positive, got {hold_days}")

    signal_dates = trading_dates[::hold_days]
    kept_signal_dates: list[pd.Timestamp] = []
    execution_dates: list[pd.Timestamp] = []

    for signal_date in signal_dates:
        idx = trading_dates.get_loc(signal_date)
        if idx + 1 >= len(trading_dates):
            continue
        kept_signal_dates.append(signal_date)
        execution_dates.append(trading_dates[idx + 1])

    return pd.DatetimeIndex(kept_signal_dates), pd.DatetimeIndex(execution_dates)
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest tests/finetune_tw/test_backtest_next_open.py::test_load_trading_calendar_uses_benchmark_dates tests/finetune_tw/test_backtest_next_open.py::test_build_signal_and_execution_dates_drops_last_anchor_without_next_day -v`

Expected: `2 passed`

- [ ] **Step 5: Commit the scaffold**

```bash
git add finetune_tw/backtest_next_open.py tests/finetune_tw/test_backtest_next_open.py
git commit -m "feat(backtest): scaffold next-open trading calendar helpers"
```

### Task 2: Build The Next-Open Portfolio Return Engine

**Files:**
- Modify: `tests/finetune_tw/test_backtest_next_open.py`
- Modify: `finetune_tw/backtest_next_open.py`

- [ ] **Step 1: Write the failing test for next-open daily return construction**

```python
def test_build_next_open_portfolio_returns_combines_gap_and_rebalance_intraday():
    import finetune_tw.backtest_next_open as bo

    trading_dates = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05"])
    execution_dates = pd.DatetimeIndex(["2024-01-03", "2024-01-05"])
    holdings = [{"A", "B"}, {"C"}]

    price_frames = {
        "A": pd.DataFrame(
            {
                "open": [100.0, 110.0, 121.0],
                "close": [110.0, 121.0, 133.1],
            },
            index=trading_dates,
        ),
        "B": pd.DataFrame(
            {
                "open": [200.0, 220.0, 198.0],
                "close": [220.0, 198.0, 217.8],
            },
            index=trading_dates,
        ),
        "C": pd.DataFrame(
            {
                "open": [50.0, 50.0, 50.0],
                "close": [50.0, 50.0, 55.0],
            },
            index=trading_dates,
        ),
    }

    period_returns, daily_returns = bo.build_next_open_portfolio_returns(
        price_frames=price_frames,
        holdings_sequence=holdings,
        execution_dates=execution_dates,
        trading_dates=trading_dates,
    )

    assert list(daily_returns.index.strftime("%Y-%m-%d")) == [
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]
    assert daily_returns.iloc[0] == pytest.approx(0.10)
    assert daily_returns.iloc[1] == pytest.approx(0.0)
    assert daily_returns.iloc[2] == pytest.approx(0.10)
    assert len(period_returns) == 1
    assert period_returns.iloc[0] == pytest.approx(0.10)
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/finetune_tw/test_backtest_next_open.py::test_build_next_open_portfolio_returns_combines_gap_and_rebalance_intraday -v`

Expected: `FAIL` with `AttributeError: module 'finetune_tw.backtest_next_open' has no attribute 'build_next_open_portfolio_returns'`

- [ ] **Step 3: Implement the minimal next-open return engine**

```python
def _mean_symbol_return(
    price_frames: dict[str, pd.DataFrame],
    symbols: set[str],
    numerator_date: pd.Timestamp,
    numerator_col: str,
    denominator_date: pd.Timestamp,
    denominator_col: str,
) -> float | None:
    values: list[float] = []
    for sym in symbols:
        frame = price_frames.get(sym)
        if frame is None:
            continue
        if numerator_date not in frame.index or denominator_date not in frame.index:
            continue
        den = float(frame.loc[denominator_date, denominator_col])
        if den == 0.0:
            continue
        num = float(frame.loc[numerator_date, numerator_col])
        values.append(num / den - 1.0)
    if not values:
        return None
    return float(np.mean(values))


def build_next_open_portfolio_returns(
    price_frames: dict[str, pd.DataFrame],
    holdings_sequence: list[set[str]],
    execution_dates: pd.DatetimeIndex,
    trading_dates: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    if len(holdings_sequence) != len(execution_dates):
        raise ValueError(
            "holdings_sequence and execution_dates must have the same length."
        )
    if len(execution_dates) < 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    daily_values: list[float] = []
    daily_index: list[pd.Timestamp] = []
    period_values: list[float] = []
    period_index: list[pd.Timestamp] = []

    first_intraday = _mean_symbol_return(
        price_frames,
        holdings_sequence[0],
        execution_dates[0],
        "close",
        execution_dates[0],
        "open",
    )
    if first_intraday is not None:
        daily_values.append(first_intraday)
        daily_index.append(execution_dates[0])

    for i in range(len(execution_dates) - 1):
        current_exec = execution_dates[i]
        next_exec = execution_dates[i + 1]
        current_holdings = holdings_sequence[i]
        next_holdings = holdings_sequence[i + 1]
        period_return = _mean_symbol_return(
            price_frames,
            current_holdings,
            next_exec,
            "open",
            current_exec,
            "open",
        )

        interior_dates = trading_dates[
            (trading_dates > current_exec) & (trading_dates < next_exec)
        ]
        prev_date = current_exec
        for date in interior_dates:
            close_to_close = _mean_symbol_return(
                price_frames,
                current_holdings,
                date,
                "close",
                prev_date,
                "close",
            )
            if close_to_close is not None:
                daily_values.append(close_to_close)
                daily_index.append(date)
            prev_date = date

        gap = _mean_symbol_return(
            price_frames,
            current_holdings,
            next_exec,
            "open",
            prev_date,
            "close",
        )
        intraday = _mean_symbol_return(
            price_frames,
            next_holdings,
            next_exec,
            "close",
            next_exec,
            "open",
        )
        if gap is not None and intraday is not None:
            daily_values.append((1.0 + gap) * (1.0 + intraday) - 1.0)
            daily_index.append(next_exec)

        if period_return is not None:
            period_values.append(period_return)
            period_index.append(current_exec)

    return (
        pd.Series(period_values, index=pd.DatetimeIndex(period_index), dtype=float),
        pd.Series(daily_values, index=pd.DatetimeIndex(daily_index), dtype=float),
    )
```

- [ ] **Step 4: Run the targeted test to verify it passes**

Run: `pytest tests/finetune_tw/test_backtest_next_open.py::test_build_next_open_portfolio_returns_combines_gap_and_rebalance_intraday -v`

Expected: `1 passed`

Self-review coverage for this task must confirm:
- rebalance-day daily return is `old holdings overnight gap` combined with `new holdings same-day intraday`
- `hold_days` means full trading sessions after entry open
- helper `period_returns` stay aligned with the legacy helper convention: one outgoing-holdings interval return from current execution open to next execution open

- [ ] **Step 5: Commit the return engine**

```bash
git add finetune_tw/backtest_next_open.py tests/finetune_tw/test_backtest_next_open.py
git commit -m "feat(backtest): add next-open return engine"
```

### Task 3: Wire The End-To-End Runner, Output Suffixes, And CLI

**Files:**
- Modify: `tests/finetune_tw/test_backtest_next_open.py`
- Modify: `finetune_tw/backtest_next_open.py`

- [ ] **Step 1: Write the failing integration-style test for outputs and schema**

```python
from types import SimpleNamespace


def _seed_runner_db(tmp_path) -> str:
    db_path = str(tmp_path / "runner.db")
    init_db(db_path)

    dates = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"]
    )
    benchmark = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": [1000.0, 1002.0, 1004.0, 1006.0, 1008.0],
            "high": [1001.0, 1003.0, 1005.0, 1007.0, 1009.0],
            "low": [999.0, 1001.0, 1003.0, 1005.0, 1007.0],
            "close": [1000.5, 1002.5, 1004.5, 1006.5, 1008.5],
            "volume": [1_000.0] * 5,
            "amount": [1_000_500.0, 1_002_500.0, 1_004_500.0, 1_006_500.0, 1_008_500.0],
        }
    )
    upsert_prices(db_path, "^TWII", benchmark)

    for symbol, opens, closes in [
        ("1101.TW", [100.0, 101.0, 102.0, 104.0, 106.0], [101.0, 102.0, 104.0, 106.0, 108.0]),
        ("1216.TW", [90.0, 89.0, 88.0, 87.0, 86.0], [89.0, 88.0, 87.0, 86.0, 85.0]),
    ]:
        frame = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "open": opens,
                "high": [x * 1.01 for x in opens],
                "low": [x * 0.99 for x in opens],
                "close": closes,
                "volume": [2_000.0] * 5,
                "amount": [close * 2_000.0 for close in closes],
            }
        )
        upsert_prices(db_path, symbol, frame)

    return db_path


def test_run_backtest_next_open_saves_suffix_outputs_and_schema(tmp_path, monkeypatch):
    import finetune_tw.backtest_next_open as bo

    db_path = _seed_runner_db(tmp_path)
    cfg = Config(
        db_path=db_path,
        output_dir=str(tmp_path),
        exp_name="next-open-test",
        lookback_window=3,
        pred_len=5,
        top_k=1,
        hold_days=2,
        test_start_date="2024-01-01",
        benchmark_symbol="^TWII",
    )

    monkeypatch.setattr(bo, "build_model_specs", lambda cfg: {"round0": SimpleNamespace(label="Round 0")})
    monkeypatch.setattr(bo, "load_predictor_from_spec", lambda spec, cfg: object())
    monkeypatch.setattr(bo, "_today", lambda: pd.Timestamp("2024-01-09"))

    def fake_compute_raw_signals(predictor, cfg, rebal_dates, pred_len, symbols):
        del predictor, pred_len
        assert list(rebal_dates.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-05"]
        assert symbols == ["1101.TW", "1216.TW"]
        return {
            "2024-01-02": {
                "1101.TW": pd.Series([0.01, 0.04]),
                "1216.TW": pd.Series([-0.01, -0.02]),
            },
            "2024-01-05": {
                "1101.TW": pd.Series([0.02, 0.03]),
                "1216.TW": pd.Series([0.04, 0.05]),
            },
        }

    monkeypatch.setattr(bo, "compute_raw_signals", fake_compute_raw_signals)

    out_path = bo.run_backtest_next_open(cfg, "round0", [2])

    out_dir = tmp_path / cfg.exp_name
    assert out_path.name == "backtest_returns_round0_next_open.json"
    assert (out_dir / "backtest_round0_next_open.png").exists()

    payload = pd.read_json(out_path)
    del payload

    import json
    data = json.loads(out_path.read_text())
    assert set(["model_key", "model_label", "test_start", "test_end", "top_k", "hold_variants", "benchmark"]).issubset(data)
    assert "2" in data["hold_variants"]
    assert len(data["hold_variants"]["2"]["dates"]) == len(data["hold_variants"]["2"]["daily_returns"])
```

- [ ] **Step 2: Run the targeted integration test to verify it fails**

Run: `pytest tests/finetune_tw/test_backtest_next_open.py::test_run_backtest_next_open_saves_suffix_outputs_and_schema -v`

Expected: `FAIL` with `AttributeError` for missing `_today` or `run_backtest_next_open`

- [ ] **Step 3: Implement the end-to-end runner, output suffixes, chart save path, and CLI**

```python
_HOLD_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]
_BM_COLOR = "#9E9E9E"
_DD_ALPHA = 0.18


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _load_price_frames(
    cfg: Config,
    symbols: list[str],
    end: str,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=cfg.test_start_date, end=end)
        if df.empty:
            continue
        frame = df[["date", "open", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frames[sym] = frame.set_index("date").sort_index()
    return frames


def plot_backtest_next_open_results(data: dict, out_dir: Path) -> Path:
    hold_keys = sorted(data["hold_variants"], key=int)
    bm = data["benchmark"]
    bm_dates = pd.DatetimeIndex(bm["dates"])
    bm_cum = (1 + pd.Series(bm["daily_returns"], index=bm_dates)).cumprod()
    colors = (_HOLD_COLORS * 4)[: len(hold_keys)]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Backtest — {data['model_label']}  ({data['test_start']} → {data['test_end']}  top-K={data['top_k']})",
        fontsize=13,
        fontweight="bold",
    )
    gs = fig.add_gridspec(2, max(len(hold_keys), 2), hspace=0.38, wspace=0.32)

    ax_cum = fig.add_subplot(gs[0, : len(hold_keys) // 2 + 1])
    ax_cum.plot(
        bm_cum.index,
        bm_cum.values,
        color=_BM_COLOR,
        lw=1.5,
        linestyle="--",
        label=f"^TWII  Sharpe={bm['metrics']['sharpe']:.2f}",
    )
    for hk, color in zip(hold_keys, colors):
        variant = data["hold_variants"][hk]
        dr = pd.Series(variant["daily_returns"], index=pd.DatetimeIndex(variant["dates"]))
        cum = (1 + dr).cumprod()
        metrics = variant["metrics"]
        ax_cum.plot(
            cum.index,
            cum.values,
            color=color,
            lw=1.8,
            label=f"hold={hk}d  Sharpe={metrics['sharpe']:.2f}  Ann={metrics['annualised_return']:.1%}",
        )
    ax_cum.set_ylabel("Cumulative Return")
    ax_cum.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax_cum.axhline(1, color="black", lw=0.6, ls=":")
    ax_cum.legend(fontsize=7.5, loc="upper left")

    ax_bar = fig.add_subplot(gs[0, len(hold_keys) // 2 + 1 :])
    metric_names = ["Ann Return", "Sharpe", "Max DD"]
    x = np.arange(len(metric_names))
    bar_w = 0.8 / (len(hold_keys) + 1)
    for i, (hk, color) in enumerate(zip(hold_keys, colors)):
        metrics = data["hold_variants"][hk]["metrics"]
        values = [metrics["annualised_return"], metrics["sharpe"] / 3, -metrics["max_drawdown"]]
        offset = (i - len(hold_keys) / 2) * bar_w
        ax_bar.bar(x + offset, values, bar_w, color=color, label=f"hold={hk}d", alpha=0.85)
    ax_bar.axhline(0, color="black", lw=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, fontsize=8)
    ax_bar.legend(fontsize=8)

    for j, (hk, color) in enumerate(zip(hold_keys, colors)):
        ax_dd = fig.add_subplot(gs[1, j])
        variant = data["hold_variants"][hk]
        dr = pd.Series(variant["daily_returns"], index=pd.DatetimeIndex(variant["dates"]))
        cum = (1 + dr).cumprod()
        dd = (cum.cummax() - cum) / cum.cummax()
        ax_dd.fill_between(dd.index, -dd.values, 0, color=color, alpha=_DD_ALPHA)
        ax_dd.plot(dd.index, -dd.values, color=color, lw=1)
        ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax_dd.set_ylim(bottom=-1)

    out_path = out_dir / f"backtest_{data['model_key']}_next_open.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_backtest_next_open(
    cfg: Config,
    model_key: str,
    hold_days_list: list[int],
) -> Path:
    specs = build_model_specs(cfg)
    if model_key not in specs:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(specs)}")
    spec = specs[model_key]

    test_end = str(_today().date())
    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    trading_dates = _load_trading_calendar(cfg, end=test_end)
    max_hold = max(hold_days_list)
    min_hold = min(hold_days_list)
    fine_signal_dates, fine_execution_dates = _build_signal_and_execution_dates(
        trading_dates,
        hold_days=min_hold,
    )

    predictor = load_predictor_from_spec(spec, cfg)
    raw_preds = compute_raw_signals(predictor, cfg, fine_signal_dates, max_hold, symbols)
    del predictor
    torch.cuda.empty_cache()

    price_frames = _load_price_frames(cfg, symbols, end=test_end)
    bm_df = query_symbol(cfg.db_path, cfg.benchmark_symbol, start=cfg.test_start_date, end=test_end)
    bm_daily = pd.Series(
        bm_df["close"].values,
        index=pd.DatetimeIndex(pd.to_datetime(bm_df["date"])),
    ).pct_change().dropna()

    hold_variants: dict[str, dict] = {}
    for hd in hold_days_list:
        step = hd // min_hold
        variant_signal_dates = fine_signal_dates[::step]
        variant_execution_dates = fine_execution_dates[::step]
        holdings = signals_to_holdings(
            raw_preds,
            variant_signal_dates,
            hd,
            cfg.top_k,
            cfg.min_signal_threshold,
        )
        _, dr = build_next_open_portfolio_returns(
            price_frames=price_frames,
            holdings_sequence=holdings,
            execution_dates=variant_execution_dates,
            trading_dates=trading_dates,
        )
        metrics = compute_metrics(dr)
        hold_variants[str(hd)] = {
            "dates": [d.strftime("%Y-%m-%d") for d in dr.index],
            "daily_returns": dr.tolist(),
            "metrics": metrics,
        }

    out = {
        "model_key": model_key,
        "model_label": spec.label,
        "test_start": cfg.test_start_date,
        "test_end": test_end,
        "top_k": cfg.top_k,
        "hold_variants": hold_variants,
        "benchmark": {
            "dates": [d.strftime("%Y-%m-%d") for d in bm_daily.index],
            "daily_returns": bm_daily.tolist(),
            "metrics": compute_metrics(bm_daily),
        },
    }

    out_dir = Path(cfg.output_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_returns_{model_key}_next_open.json"
    out_path.write_text(json.dumps(out, indent=2))
    plot_backtest_next_open_results(out, out_dir)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--model", required=True, choices=["pretrained", "round0", "round1", "round2"])
    parser.add_argument("--hold_days_list", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--test_start", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.top_k:
        cfg.top_k = args.top_k
    if args.test_start:
        cfg.test_start_date = args.test_start
    if args.threshold is not None:
        cfg.min_signal_threshold = args.threshold

    run_backtest_next_open(cfg, args.model, args.hold_days_list)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the targeted integration test to verify it passes**

Run: `pytest tests/finetune_tw/test_backtest_next_open.py::test_run_backtest_next_open_saves_suffix_outputs_and_schema -v`

Expected: `1 passed`

- [ ] **Step 5: Commit the end-to-end module**

```bash
git add finetune_tw/backtest_next_open.py tests/finetune_tw/test_backtest_next_open.py
git commit -m "feat(backtest): add next-open backtest runner"
```

### Task 4: Verify The Full Test Slice And Clean Up Edge Cases

**Files:**
- Modify: `tests/finetune_tw/test_backtest_next_open.py`
- Modify: `finetune_tw/backtest_next_open.py`

- [ ] **Step 1: Add the final edge-case tests**

```python
def test_build_next_open_portfolio_returns_returns_empty_series_without_two_entries():
    import finetune_tw.backtest_next_open as bo

    trading_dates = pd.DatetimeIndex(["2024-01-03"])
    execution_dates = pd.DatetimeIndex(["2024-01-03"])
    price_frames = {
        "A": pd.DataFrame({"open": [100.0], "close": [101.0]}, index=trading_dates)
    }

    period_returns, daily_returns = bo.build_next_open_portfolio_returns(
        price_frames=price_frames,
        holdings_sequence=[{"A"}],
        execution_dates=execution_dates,
        trading_dates=trading_dates,
    )

    assert period_returns.empty
    assert daily_returns.empty


def test_run_backtest_next_open_raises_when_benchmark_calendar_missing(tmp_path):
    import finetune_tw.backtest_next_open as bo

    db_path = str(tmp_path / "empty.db")
    init_db(db_path)
    cfg = Config(
        db_path=db_path,
        output_dir=str(tmp_path),
        exp_name="empty-next-open",
        test_start_date="2024-01-01",
        benchmark_symbol="^TWII",
    )

    with pytest.raises(ValueError, match="No benchmark rows found"):
        bo._load_trading_calendar(cfg, end="2024-01-31")


def test_run_backtest_next_open_raises_when_hold_variant_has_no_realized_returns(
    tmp_path,
    monkeypatch,
):
    import finetune_tw.backtest_next_open as bo

    db_path = _seed_runner_db(tmp_path)
    cfg = Config(
        db_path=db_path,
        output_dir=str(tmp_path),
        exp_name="empty-returns-next-open",
        lookback_window=3,
        pred_len=5,
        top_k=1,
        hold_days=5,
        test_start_date="2024-01-01",
        benchmark_symbol="^TWII",
    )

    monkeypatch.setattr(bo, "build_model_specs", lambda cfg: {"round0": SimpleNamespace(label="Round 0")})
    monkeypatch.setattr(bo, "load_predictor_from_spec", lambda spec, cfg: object())
    monkeypatch.setattr(bo, "_today", lambda: pd.Timestamp("2024-01-09"))
    monkeypatch.setattr(
        bo,
        "compute_raw_signals",
        lambda predictor, cfg, rebal_dates, pred_len, symbols: {
            "2024-01-02": {"1101.TW": pd.Series([0.01] * 5)}
        },
    )

    with pytest.raises(ValueError, match="No realized daily returns for hold_days=5"):
        bo.run_backtest_next_open(cfg, "round0", [5])
```

- [ ] **Step 2: Run the new edge-case tests to verify they fail for the right reason**

Run: `pytest tests/finetune_tw/test_backtest_next_open.py::test_build_next_open_portfolio_returns_returns_empty_series_without_two_entries tests/finetune_tw/test_backtest_next_open.py::test_run_backtest_next_open_raises_when_benchmark_calendar_missing tests/finetune_tw/test_backtest_next_open.py::test_run_backtest_next_open_raises_when_hold_variant_has_no_realized_returns -v`

Expected: the empty-series test passes if Task 2 already handled `len(execution_dates) < 2` correctly. The new runner-level test should fail because `run_backtest_next_open()` still passes an empty `daily_returns` series into `compute_metrics()` instead of raising a clear `ValueError`.

- [ ] **Step 3: Raise a clear runner-level error when a hold variant has no realized returns**

```python
for hd in hold_days_list:
    step = hd // min_hold
    variant_signal_dates = fine_signal_dates[::step]
    variant_execution_dates = fine_execution_dates[::step]
    holdings = signals_to_holdings(
        raw_preds,
        variant_signal_dates,
        hd,
        cfg.top_k,
        cfg.min_signal_threshold,
    )
    _, dr = build_next_open_portfolio_returns(
        price_frames=price_frames,
        holdings_sequence=holdings,
        execution_dates=variant_execution_dates,
        trading_dates=trading_dates,
    )
    if dr.empty:
        raise ValueError(
            f"No realized daily returns for hold_days={hd}. "
            "Check trading calendar coverage and execution dates."
        )
    metrics = compute_metrics(dr)
    hold_variants[str(hd)] = {
        "dates": [d.strftime("%Y-%m-%d") for d in dr.index],
        "daily_returns": dr.tolist(),
        "metrics": metrics,
    }
```

- [ ] **Step 4: Run the full next-open test slice and baseline backtest tests**

Run: `pytest tests/finetune_tw/test_backtest.py tests/finetune_tw/test_backtest_next_open.py -v`

Expected: all tests `PASS`

- [ ] **Step 5: Commit the verified edge-case coverage**

```bash
git add finetune_tw/backtest_next_open.py tests/finetune_tw/test_backtest_next_open.py
git commit -m "test(backtest): cover next-open edge cases"
```

## Self-Review

### Spec coverage

- Separate module and unchanged old backtest: covered by Tasks 1 and 3
- Real benchmark trading calendar: covered by Task 1
- Same signal-generation path: covered by Task 3 through `compute_raw_signals` reuse
- `T close -> T+1 open` execution timing: covered by Tasks 1 and 2
- Next-open daily return construction including rebalance-day gap/intraday split: covered by Task 2
- `hold_days` means full held trading sessions after entry open: covered by Task 2’s corrected expected daily return and Task 3’s execution-date wiring
- Same CLI shape and JSON schema: covered by Task 3
- `_next_open` output suffixes: covered by Task 3
- Final anchor without a following trading day: covered by Tasks 1 and 4
- Empty realized-return handling with a clear error: covered by Task 4

### Placeholder scan

- No `TBD`, `TODO`, or “implement later” placeholders remain
- Every code-writing step contains an explicit code block
- Every verification step contains an explicit command and expected result

### Type consistency

- Helper names used consistently:
  - `_load_trading_calendar`
  - `_build_signal_and_execution_dates`
  - `_mean_symbol_return`
  - `build_next_open_portfolio_returns`
  - `run_backtest_next_open`
  - `plot_backtest_next_open_results`
- `price_frames` is consistently a `dict[str, pd.DataFrame]` indexed by `date`
- `execution_dates` is consistently a `pd.DatetimeIndex`
- `holdings_sequence` stays aligned one-to-one with `execution_dates`
