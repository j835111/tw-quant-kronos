#!/usr/bin/env python3
"""
Unit test for rank_h decoupling in signals_to_holdings.
Run with: python3 finetune_tw/test_rank_h.py
"""
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune_tw.backtest import signals_to_holdings


def test_rank_h_decoupling():
    """Test that rank_h decouples ranking horizon from hold period."""

    # Create synthetic raw_preds for a single date with 5 symbols
    # Each symbol has 10 days of predicted returns
    test_date = "2025-01-15"

    # Design: symbol A has highest 1-day return but low 5-day return
    #         symbol B has low 1-day return but highest 5-day return
    #         symbols C, D, E fill in the middle
    raw_preds = {
        test_date: {
            "SYMB_A": pd.Series([0.05, 0.02, 0.00, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07]),  # h=1: 0.05, h=5: -0.02
            "SYMB_B": pd.Series([-0.01, 0.00, 0.01, 0.02, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]),          # h=1: -0.01, h=5: 0.08
            "SYMB_C": pd.Series([0.02, 0.01, 0.01, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),           # moderate in both
            "SYMB_D": pd.Series([0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),           # flat
            "SYMB_E": pd.Series([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),           # zero
        }
    }

    test_dates = pd.DatetimeIndex([test_date])

    # Test 1: rank by horizon h=1 with hold_days=5
    print("Test 1: rank_h=1, hold_days=5")
    holdings_rank_h1 = signals_to_holdings(raw_preds, test_dates, hold_days=5, top_k=2, rank_h=1)
    print(f"  Holdings (rank_h=1): {holdings_rank_h1[0]}")
    assert len(holdings_rank_h1[0]) <= 2, "Should have at most 2 holdings"

    # Test 2: rank by horizon h=5 with hold_days=5
    print("Test 2: rank_h=5, hold_days=5")
    holdings_rank_h5 = signals_to_holdings(raw_preds, test_dates, hold_days=5, top_k=2, rank_h=5)
    print(f"  Holdings (rank_h=5): {holdings_rank_h5[0]}")
    assert len(holdings_rank_h5[0]) <= 2, "Should have at most 2 holdings"

    # Test 3: rank_h=None should equal rank_h=hold_days
    print("Test 3: rank_h=None should equal rank_h=hold_days")
    holdings_rank_none = signals_to_holdings(raw_preds, test_dates, hold_days=5, top_k=2, rank_h=None)
    print(f"  Holdings (rank_h=None): {holdings_rank_none[0]}")
    assert holdings_rank_none[0] == holdings_rank_h5[0], "rank_h=None should equal rank_h=hold_days"

    # Test 4: Verify that rank_h=1 and rank_h=5 produce different results
    print("Test 4: Verify rank_h=1 != rank_h=5")
    if holdings_rank_h1[0] != holdings_rank_h5[0]:
        print(f"  ✓ PASS: Rankings differ as expected")
        print(f"    rank_h=1 picks: {holdings_rank_h1[0]}")
        print(f"    rank_h=5 picks: {holdings_rank_h5[0]}")
    else:
        print(f"  ✗ FAIL: Rankings should differ but are the same: {holdings_rank_h1[0]}")
        return False

    # Test 5: Verify SYMB_A is in top-2 for rank_h=1 (it has highest 1-day return)
    print("Test 5: SYMB_A in rank_h=1 top-2")
    if "SYMB_A" in holdings_rank_h1[0]:
        print(f"  ✓ PASS: SYMB_A selected for rank_h=1")
    else:
        print(f"  ✗ FAIL: SYMB_A should be in rank_h=1 top-2, but got: {holdings_rank_h1[0]}")
        return False

    # Test 6: Verify SYMB_B is in top-2 for rank_h=5 (it has highest 5-day return)
    print("Test 6: SYMB_B in rank_h=5 top-2")
    if "SYMB_B" in holdings_rank_h5[0]:
        print(f"  ✓ PASS: SYMB_B selected for rank_h=5")
    else:
        print(f"  ✗ FAIL: SYMB_B should be in rank_h=5 top-2, but got: {holdings_rank_h5[0]}")
        return False

    print("\n" + "="*60)
    print("All tests PASSED!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_rank_h_decoupling()
    sys.exit(0 if success else 1)
