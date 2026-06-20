import numpy as np
import pandas as pd
import pytest

from finetune_tw.ic_validation import (
    EarlyStopper,
    mean_cross_sectional_ic,
    pick_val_dates,
    pick_val_universe,
    rank_ic,
)


def test_rank_ic_perfect_positive():
    assert rank_ic([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_rank_ic_perfect_negative():
    assert rank_ic([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_rank_ic_too_few_points_is_nan():
    assert np.isnan(rank_ic([1, 2], [3, 4]))


def test_rank_ic_zero_variance_is_nan():
    assert np.isnan(rank_ic([1, 1, 1, 1], [1, 2, 3, 4]))


def test_mean_cross_sectional_ic_averages_groups():
    per_group = {
        "d1": ([1, 2, 3, 4], [10, 20, 30, 40]),
        "d2": ([1, 2, 3, 4], [40, 30, 20, 10]),
    }
    assert mean_cross_sectional_ic(per_group) == pytest.approx(0.0)


def test_mean_cross_sectional_ic_skips_nan_groups():
    per_group = {
        "d1": ([1, 2, 3, 4], [10, 20, 30, 40]),
        "d2": ([1, 1], [2, 3]),
    }
    assert mean_cross_sectional_ic(per_group) == pytest.approx(1.0)


def test_pick_val_universe_deterministic_and_sized():
    syms = [f"{i:04d}" for i in range(1000)]
    a = pick_val_universe(syms, 150, seed=42)
    b = pick_val_universe(syms, 150, seed=42)
    assert a == b
    assert len(a) == 150
    assert len(set(a)) == 150


def test_pick_val_universe_returns_all_if_small():
    syms = ["A", "B", "C"]
    assert pick_val_universe(syms, 150) == ["A", "B", "C"]


def test_pick_val_dates_count_and_bounds():
    dates = pick_val_dates("2024-01-01", "2024-06-30", 8)
    assert len(dates) <= 8
    assert dates == sorted(dates)
    assert dates[0] >= pd.Timestamp("2024-01-01")
    assert dates[-1] <= pd.Timestamp("2024-06-30")


def test_early_stopper_first_value_is_best():
    es = EarlyStopper(patience=2, mode="max")
    is_best, stop = es.update(0.1)
    assert is_best and not stop
    assert es.best == 0.1


def test_early_stopper_improvement_resets_patience():
    es = EarlyStopper(patience=2, mode="max")
    es.update(0.1)
    is_best, stop = es.update(0.2)
    assert is_best and not stop


def test_early_stopper_stops_after_patience():
    es = EarlyStopper(patience=2, mode="max")
    es.update(0.3)
    assert es.update(0.2) == (False, False)
    assert es.update(0.1) == (False, False)
    assert es.update(0.1) == (False, True)


def test_early_stopper_nan_counts_as_no_improvement():
    es = EarlyStopper(patience=1, mode="max")
    es.update(0.3)
    assert es.update(float("nan")) == (False, False)
    assert es.update(float("nan")) == (False, True)
