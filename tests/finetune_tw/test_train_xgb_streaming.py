import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

from finetune_tw.lambdarank_ic import lambdarank_ic_objective
from finetune_tw.train_xgb_lambdarank import build_group_sizes
from finetune_tw.train_xgb_streaming import (
    feature_set_columns,
    grouped_ndcg_at_k,
    grouped_top_k_excess,
    load_val_matrix,
    parallel_lambdarank_ic_objective,
    resolve_date_filter,
    resolve_multi_date_filter,
    scan_group_sizes,
    train_streaming,
    _parse_range_arg,
)


def _make_synthetic_df(n_dates=6, n_symbols=20, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = f"2024-01-{d + 1:02d}"
        true_factor = rng.normal(size=n_symbols)
        for s in range(n_symbols):
            emb = rng.normal(size=8)
            emb[0] += true_factor[s]
            label = true_factor[s] + rng.normal(scale=0.1)
            row = {"date": date, "symbol": f"S{s}", "label": label,
                   "feat_ma5_dist": rng.normal(), "feat_momentum_10": rng.normal()}
            row.update({f"emb_{k}": float(v) for k, v in enumerate(emb)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_feature_set_columns():
    cols = ["date", "symbol", "label", "emb_1", "emb_0", "emb_10",
            "feat_ma5_dist", "feat_ma5_dist_cs_rank", "feat_momentum_10"]
    assert feature_set_columns(cols, "emb") == ["emb_0", "emb_1", "emb_10"]
    assert feature_set_columns(cols, "raw") == [
        "feat_ma5_dist", "feat_ma5_dist_cs_rank", "feat_momentum_10"]
    assert feature_set_columns(cols, "full") == [
        "emb_0", "emb_1", "emb_10",
        "feat_ma5_dist", "feat_ma5_dist_cs_rank", "feat_momentum_10"]


def test_grouped_top_tail_metrics_reward_better_ordering():
    group_sizes = [5, 5]
    labels = np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=np.float32)
    perfect = np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=np.float32)
    bad = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1], dtype=np.float32)

    assert grouped_top_k_excess(perfect, labels, group_sizes, top_k=2) > grouped_top_k_excess(
        bad, labels, group_sizes, top_k=2
    )
    assert grouped_ndcg_at_k(perfect, labels, group_sizes, top_k=2) == pytest.approx(1.0)
    assert grouped_ndcg_at_k(perfect, labels, group_sizes, top_k=2) > grouped_ndcg_at_k(
        bad, labels, group_sizes, top_k=2
    )


def test_parallel_objective_matches_serial():
    df = _make_synthetic_df(n_dates=4, n_symbols=15, seed=3)
    groups = build_group_sizes(df)
    preds = np.random.default_rng(1).normal(size=len(df))

    class FakeDMatrix:
        def get_label(self):
            return df["label"].values

    g_serial, h_serial = lambdarank_ic_objective(groups)(preds, FakeDMatrix())
    obj = parallel_lambdarank_ic_objective(groups, n_threads=4)
    try:
        g_par, h_par = obj(preds, FakeDMatrix())
    finally:
        obj.close()
    np.testing.assert_allclose(g_par, g_serial)
    np.testing.assert_allclose(h_par, h_serial)


def test_scan_group_sizes_filters_and_validates(tmp_path):
    df = _make_synthetic_df(n_dates=4, n_symbols=10)
    path = tmp_path / "t.parquet"
    df.to_parquet(path, index=False)
    assert scan_group_sizes(path, None) == [10, 10, 10, 10]
    assert scan_group_sizes(path, {"2024-01-02", "2024-01-04"}) == [10, 10]

    shuffled = pd.concat([df.iloc[:5], df.iloc[15:], df.iloc[5:15]])  # splits date 1
    shuffled.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="non-contiguous"):
        scan_group_sizes(path, None)


def test_scan_group_sizes_streams_dates_without_parquetfile_read(tmp_path, monkeypatch):
    df = _make_synthetic_df(n_dates=4, n_symbols=10)
    path = tmp_path / "t.parquet"
    df.to_parquet(path, index=False)

    def fail_read(self, *args, **kwargs):
        raise RuntimeError("ParquetFile.read should not be called")

    monkeypatch.setattr("pyarrow.parquet.ParquetFile.read", fail_read)

    assert scan_group_sizes(path, None) == [10, 10, 10, 10]
    assert scan_group_sizes(path, {"2024-01-02", "2024-01-04"}) == [10, 10]


def test_load_val_matrix_multi_file_avoids_dmatrix_get_data(tmp_path, monkeypatch):
    val_a = _make_synthetic_df(n_dates=2, n_symbols=20, seed=12)
    val_b = _make_synthetic_df(n_dates=3, n_symbols=20, seed=13)
    val_a_path = tmp_path / "val_a.parquet"
    val_b_path = tmp_path / "val_b.parquet"
    val_a.to_parquet(val_a_path, index=False)
    val_b.to_parquet(val_b_path, index=False)

    def fail_get_data(self):
        raise RuntimeError("get_data should not be called")

    monkeypatch.setattr(xgb.DMatrix, "get_data", fail_get_data)

    dval, groups = load_val_matrix([val_a_path, val_b_path], ["emb_0"], {"2024-01-02"})
    assert groups == [40]
    assert dval.num_row() == 40
    assert dval.num_col() == 1
    np.testing.assert_array_equal(dval.get_label(), np.concatenate([
        val_a.loc[val_a["date"] == "2024-01-02", "label"].to_numpy(dtype=np.float32),
        val_b.loc[val_b["date"] == "2024-01-02", "label"].to_numpy(dtype=np.float32),
    ]))


def test_load_val_matrix_merges_group_across_file_boundary(tmp_path):
    val = _make_synthetic_df(n_dates=3, n_symbols=20, seed=12)
    val_a = pd.concat([
        val.loc[val["date"] == "2024-01-01"],
        val.loc[val["date"] == "2024-01-02"].iloc[:7],
    ])
    val_b = pd.concat([
        val.loc[val["date"] == "2024-01-02"].iloc[7:],
        val.loc[val["date"] == "2024-01-03"],
    ])
    val_a_path = tmp_path / "val_a.parquet"
    val_b_path = tmp_path / "val_b.parquet"
    val_a.to_parquet(val_a_path, index=False)
    val_b.to_parquet(val_b_path, index=False)

    dval, groups = load_val_matrix([val_a_path, val_b_path], ["emb_0"], None)

    assert groups == [20, 20, 20]
    assert dval.num_row() == 60
    np.testing.assert_array_equal(dval.get_label(), val["label"].to_numpy(dtype=np.float32))


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


def test_train_streaming_closes_objective_after_training_error(tmp_path, monkeypatch):
    train_df = _make_synthetic_df(n_dates=4, n_symbols=10, seed=1)
    val_df = _make_synthetic_df(n_dates=2, n_symbols=10, seed=2)
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    class FakeObjective:
        def __init__(self):
            self.closed = False

        def __call__(self, preds, dtrain):
            raise AssertionError("objective should not be called in this test")

        def close(self):
            self.closed = True

    fake_obj = FakeObjective()

    def fake_objective_factory(*args, **kwargs):
        return fake_obj

    def fail_train(*args, **kwargs):
        raise RuntimeError("train failed")

    monkeypatch.setattr(
        "finetune_tw.train_xgb_streaming.parallel_lambdarank_ic_objective",
        fake_objective_factory,
    )
    monkeypatch.setattr(xgb, "train", fail_train)

    with pytest.raises(RuntimeError, match="train failed"):
        train_streaming(
            train_path,
            val_path,
            feature_set="emb",
            keep_dates=None,
            num_boost_round=5,
            early_stopping_rounds=2,
            n_threads=2,
        )

    assert fake_obj.closed is True


def test_train_streaming_improves_rank_ic(tmp_path):
    train_df = _make_synthetic_df(n_dates=10, n_symbols=30, seed=1)
    val_df = _make_synthetic_df(n_dates=4, n_symbols=30, seed=2)
    train_path, val_path = tmp_path / "train.parquet", tmp_path / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    booster, summary = train_streaming(
        train_path, val_path, feature_set="emb", keep_dates=None,
        num_boost_round=50, early_stopping_rounds=10, n_threads=2,
    )
    assert summary["n_features"] == 8
    assert summary["train_rows"] == 300
    assert summary["best_val_rank_ic"] > 0.5  # emb_0 carries the signal

    # raw-only sees only noise features -> should not reach the emb-only IC
    _, raw_summary = train_streaming(
        train_path, val_path, feature_set="raw", keep_dates=None,
        num_boost_round=20, early_stopping_rounds=5, n_threads=2,
    )
    assert raw_summary["n_features"] == 2
    assert raw_summary["best_val_rank_ic"] < summary["best_val_rank_ic"]


def test_train_streaming_accepts_multiple_validation_parquets(tmp_path):
    train_df = _make_synthetic_df(n_dates=8, n_symbols=20, seed=11)
    val_a = _make_synthetic_df(n_dates=2, n_symbols=20, seed=12)
    val_b = _make_synthetic_df(n_dates=3, n_symbols=20, seed=13)
    val_b = val_b.copy()
    val_b["date"] = val_b["date"].map({
        "2024-01-01": "2024-01-03",
        "2024-01-02": "2024-01-04",
        "2024-01-03": "2024-01-05",
    })
    train_path = tmp_path / "train.parquet"
    val_a_path = tmp_path / "val_a.parquet"
    val_b_path = tmp_path / "val_b.parquet"
    train_df.to_parquet(train_path, index=False)
    val_a.to_parquet(val_a_path, index=False)
    val_b.to_parquet(val_b_path, index=False)

    _, summary = train_streaming(
        train_path,
        [val_a_path, val_b_path],
        feature_set="emb",
        keep_dates=None,
        num_boost_round=20,
        early_stopping_rounds=5,
        n_threads=2,
    )
    assert summary["val_rows"] == len(val_a) + len(val_b)
    assert summary["val_dates"] == 5


def test_parse_range_arg_splits_start_end():
    assert _parse_range_arg("2021-01-04:2021-06-30", "--val-range") == ("2021-01-04", "2021-06-30")


def test_parse_range_arg_rejects_malformed_value():
    with pytest.raises(ValueError, match="START:END"):
        _parse_range_arg("2021-01-04", "--val-range")
    with pytest.raises(ValueError, match="START:END"):
        _parse_range_arg("2021-01-04:2021-06-30:extra", "--val-range")


def test_resolve_multi_date_filter_unions_disjoint_ranges():
    keep_dates = resolve_multi_date_filter(
        [("2021-01-04", "2021-06-30"), ("2023-01-03", "2024-06-28")],
        trading_days={"2021-03-15", "2021-07-01", "2023-06-01", "2025-01-01"},
    )
    assert keep_dates == {"2021-03-15", "2023-06-01"}


def test_resolve_multi_date_filter_empty_ranges_returns_none():
    assert resolve_multi_date_filter([], trading_days={"2024-01-01"}) is None


def test_train_streaming_val_spans_disjoint_regimes_across_two_files(tmp_path):
    # Simulate: momentum-regime slice lives in the "train" source file, the later
    # slice lives in the "val" source file -- both get pulled into one val set.
    train_df = _make_synthetic_df(n_dates=8, n_symbols=15, seed=31)
    later_df = _make_synthetic_df(n_dates=3, n_symbols=15, seed=32)
    later_df["date"] = later_df["date"].map({
        "2024-01-01": "2024-02-01",
        "2024-01-02": "2024-02-02",
        "2024-01-03": "2024-02-03",
    })
    train_path = tmp_path / "train_src.parquet"
    later_path = tmp_path / "later_src.parquet"
    train_df.to_parquet(train_path, index=False)
    later_df.to_parquet(later_path, index=False)

    momentum_dates = {"2024-01-02", "2024-01-04"}  # a subset that lives in train_df
    later_dates = {"2024-02-01", "2024-02-03"}
    val_keep_dates = momentum_dates | later_dates

    _, summary = train_streaming(
        train_path,
        [train_path, later_path],
        feature_set="emb",
        train_keep_dates={"2024-01-01", "2024-01-03", "2024-01-05"},
        val_keep_dates=val_keep_dates,
        num_boost_round=10,
        early_stopping_rounds=3,
        n_threads=2,
    )

    assert summary["train_dates"] == 3
    assert summary["val_dates"] == 4
    assert summary["val_rows"] == 4 * 15


def test_train_streaming_uses_independent_train_and_val_date_filters(tmp_path):
    df = _make_synthetic_df(n_dates=6, n_symbols=20, seed=21)
    path = tmp_path / "all.parquet"
    df.to_parquet(path, index=False)

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
