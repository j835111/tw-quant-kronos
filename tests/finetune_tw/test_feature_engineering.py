import numpy as np
import pandas as pd
import pytest

from finetune_tw.enrich_round6_features import (
    _iter_date_blocks,
    _merge_feature_block,
    enrich_parquet,
)
from finetune_tw.feature_engineering import (
    TECH_FEATURE_COLUMNS,
    add_cross_sectional_rank_features,
    compute_technical_feature_frame,
    compute_technical_features,
    technical_feature_columns,
)


def test_compute_technical_feature_frame_matches_pointwise_function():
    n = 70
    idx = np.arange(n, dtype=np.float64)
    history = pd.DataFrame({
        "symbol": ["AAA"] * n,
        "date": pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "open": 100.0 + idx,
        "high": 101.0 + idx,
        "low": 99.0 + idx,
        "close": 100.5 + idx,
        "volume": 1000.0 + idx * 10.0,
        "amount": 10000.0 + idx * 100.0,
    })

    pointwise = compute_technical_features(history[["open", "high", "low", "close", "volume", "amount"]])
    frame = compute_technical_feature_frame(history)

    last_row = frame.iloc[-1]
    assert list(frame.columns) == ["symbol", "date", *TECH_FEATURE_COLUMNS]
    for key, expected in pointwise.items():
        assert last_row[key] == pytest.approx(expected)


def test_add_cross_sectional_rank_features_ranks_each_date_independently():
    df = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        "symbol": ["AAA", "BBB", "AAA", "BBB"],
        "feat_ma5_dist": [0.1, 0.3, 0.9, 0.2],
        "feat_momentum_3": [5.0, 1.0, 7.0, 7.0],
    })

    ranked = add_cross_sectional_rank_features(df, feature_cols=["feat_ma5_dist", "feat_momentum_3"])

    assert ranked["feat_ma5_dist_cs_rank"].tolist() == pytest.approx([0.5, 1.0, 1.0, 0.5])
    assert ranked["feat_momentum_3_cs_rank"].tolist() == pytest.approx([1.0, 0.5, 0.75, 0.75])


def test_technical_feature_columns_include_base_and_cs_rank_features():
    columns = [
        "label",
        "feat_momentum_10_cs_rank",
        "feat_ma5_dist",
        "feat_hl_spread_5",
        "feat_ma5_dist_cs_rank",
        "feat_momentum_10",
        "emb_0",
    ]

    assert technical_feature_columns(columns) == [
        "feat_ma5_dist",
        "feat_ma5_dist_cs_rank",
        "feat_momentum_10",
        "feat_momentum_10_cs_rank",
        "feat_hl_spread_5",
    ]


def test_iter_date_blocks_reassembles_split_dates_before_ranking(tmp_path):
    df = pd.DataFrame({
        "date": [
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
        ],
        "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB"],
        "feat_ma5_dist": [0.1, 0.3, 0.2, 0.9, 0.2],
    })
    path = tmp_path / "split_dates.parquet"
    df.to_parquet(path, index=False)

    blocks = list(_iter_date_blocks(str(path), keep_dates=None, batch_size=2))

    assert [date for date, _ in blocks] == ["2024-01-02", "2024-01-03"]
    assert [len(block) for _, block in blocks] == [3, 2]

    ranked_first = add_cross_sectional_rank_features(
        blocks[0][1], feature_cols=["feat_ma5_dist"]
    )
    assert ranked_first["feat_ma5_dist_cs_rank"].tolist() == pytest.approx([1 / 3, 1.0, 2 / 3])


def test_iter_date_blocks_raises_when_a_date_reappears_in_a_later_run(tmp_path):
    df = pd.DataFrame({
        "date": [
            "2024-01-02",
            "2024-01-03",
            "2024-01-02",
        ],
        "symbol": ["AAA", "AAA", "BBB"],
        "label": [0.1, 0.2, 0.3],
    })
    path = tmp_path / "nonmonotonic.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="contiguous run"):
        list(_iter_date_blocks(str(path), keep_dates=None, batch_size=10))


def test_merge_feature_block_merges_single_date_block_without_global_lookup():
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

    assert list(merged.columns) == [
        "date",
        "symbol",
        "label",
        "emb_0",
        "feat_ma5_dist",
        "feat_ma5_dist_cs_rank",
    ]
    assert merged["feat_ma5_dist"].tolist() == pytest.approx([0.01, 0.02])
    assert merged["feat_ma5_dist_cs_rank"].tolist() == pytest.approx([0.5, 1.0])


def test_enrich_parquet_recomputes_features_by_date_block_on_split_parquet_batches(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    artifact = pd.DataFrame({
        "date": [
            "2024-03-20",
            "2024-03-20",
            "2024-03-20",
            "2024-03-21",
            "2024-03-21",
        ],
        "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB"],
        "label": [0.1, 0.2, 0.3, 0.4, 0.5],
        "emb_0": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_ma5_dist": [-999.0, -999.0, -999.0, -999.0, -999.0],
    })
    artifact.to_parquet(input_path, index=False)

    history = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC"],
        "date": ["2024-03-20", "2024-03-21", "2024-03-20", "2024-03-21", "2024-03-20"],
        "open": [1.0, 1.0, 1.0, 1.0, 1.0],
        "high": [1.0, 1.0, 1.0, 1.0, 1.0],
        "low": [1.0, 1.0, 1.0, 1.0, 1.0],
        "close": [1.0, 1.0, 2.0, 3.0, 1.5],
        "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        "amount": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    query_calls: list[tuple[tuple[str, ...], str, str]] = []

    def fake_query_symbols_window(db_path: str, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        query_calls.append((tuple(symbols), start, end))
        mask = history["symbol"].isin(symbols) & history["date"].between(start, end)
        return history.loc[mask].copy()

    monkeypatch.setattr("finetune_tw.enrich_round6_features.query_symbols_window", fake_query_symbols_window)

    enrich_parquet(
        str(input_path),
        str(output_path),
        db_path="ignored.db",
        batch_size=2,
        keep_dates=None,
        buffer_days=0,
    )

    out = pd.read_parquet(output_path)

    assert query_calls == [
        (("AAA", "BBB", "CCC"), "2024-03-20", "2024-03-20"),
        (("AAA", "BBB"), "2024-03-21", "2024-03-21"),
    ]
    assert out["date"].tolist() == artifact["date"].tolist()
    assert out["symbol"].tolist() == artifact["symbol"].tolist()
    assert out["label"].tolist() == pytest.approx(artifact["label"].tolist())
    assert out["emb_0"].tolist() == pytest.approx(artifact["emb_0"].tolist())
    assert "feat_ma5_dist" in out.columns
    assert "feat_ma5_dist_cs_rank" in out.columns
    assert out["feat_ma5_dist"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])
    assert out["feat_ma5_dist_cs_rank"].tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3, 0.75, 0.75])
    assert not any(value == -999.0 for value in out["feat_ma5_dist"].tolist())


def test_enrich_parquet_does_not_depend_on_whole_history_feature_frame_materialization(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    artifact = pd.DataFrame({
        "date": [
            "2024-03-20",
            "2024-03-20",
            "2024-03-20",
            "2024-03-21",
            "2024-03-21",
        ],
        "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB"],
        "label": [0.1, 0.2, 0.3, 0.4, 0.5],
        "emb_0": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_ma5_dist": [-999.0, -999.0, -999.0, -999.0, -999.0],
    })
    artifact.to_parquet(input_path, index=False)

    history = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC"],
        "date": ["2024-03-20", "2024-03-21", "2024-03-20", "2024-03-21", "2024-03-20"],
        "open": [1.0, 1.0, 1.0, 1.0, 1.0],
        "high": [1.0, 1.0, 1.0, 1.0, 1.0],
        "low": [1.0, 1.0, 1.0, 1.0, 1.0],
        "close": [1.0, 1.0, 2.0, 3.0, 1.5],
        "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        "amount": [1.0, 1.0, 1.0, 1.0, 1.0],
    })

    def fake_query_symbols_window(db_path: str, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        mask = history["symbol"].isin(symbols) & history["date"].between(start, end)
        return history.loc[mask].copy()

    def fail_compute_technical_feature_frame(_: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("whole-history feature-frame path should not be used")

    monkeypatch.setattr("finetune_tw.enrich_round6_features.query_symbols_window", fake_query_symbols_window)
    monkeypatch.setattr(
        "finetune_tw.feature_engineering.compute_technical_feature_frame",
        fail_compute_technical_feature_frame,
    )

    enrich_parquet(
        str(input_path),
        str(output_path),
        db_path="ignored.db",
        batch_size=2,
        keep_dates=None,
        buffer_days=0,
    )

    out = pd.read_parquet(output_path)
    assert out["feat_ma5_dist"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])
    assert out["feat_ma5_dist_cs_rank"].tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3, 0.75, 0.75])


def test_enrich_parquet_drops_rows_missing_same_day_db_prices(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    artifact = pd.DataFrame({
        "date": ["2024-03-20", "2024-03-20"],
        "symbol": ["AAA", "BBB"],
        "label": [0.1, 0.2],
        "emb_0": [1.0, 2.0],
        "feat_ma5_dist": [-999.0, -999.0],
    })
    artifact.to_parquet(input_path, index=False)

    history = pd.DataFrame({
        "symbol": ["AAA"],
        "date": ["2024-03-20"],
        "open": [1.0],
        "high": [1.0],
        "low": [1.0],
        "close": [1.0],
        "volume": [1.0],
        "amount": [1.0],
    })

    def fake_query_symbols_window(db_path: str, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        mask = history["symbol"].isin(symbols) & history["date"].between(start, end)
        return history.loc[mask].copy()

    monkeypatch.setattr("finetune_tw.enrich_round6_features.query_symbols_window", fake_query_symbols_window)

    enrich_parquet(
        str(input_path),
        str(output_path),
        db_path="ignored.db",
        batch_size=2,
        keep_dates=None,
        buffer_days=0,
    )

    out = pd.read_parquet(output_path)
    log = capsys.readouterr().out
    assert "Dropping parquet rows without same-day DB prices" in log
    assert out["symbol"].tolist() == ["AAA"]
    assert out["feat_ma5_dist_cs_rank"].tolist() == pytest.approx([1.0])
