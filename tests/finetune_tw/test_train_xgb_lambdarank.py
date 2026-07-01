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
