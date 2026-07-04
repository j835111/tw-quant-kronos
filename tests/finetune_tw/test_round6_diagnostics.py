import sqlite3

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa

from finetune_tw.round6_diagnostics import (
    aggregate_period,
    iter_scored_dates,
    per_day_metrics,
    stream_scores,
    twse_trading_days,
)


@pytest.fixture()
def calendar_db(tmp_path):
    """daily_prices with: 2 clean trading days, 1 typhoon day (stocks but no benchmark),
    1 dirty benchmark day (benchmark row but almost no stocks)."""
    db_path = tmp_path / "tw.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE daily_prices (symbol TEXT, date TEXT, open REAL, high REAL,"
        " low REAL, close REAL, volume REAL, amount REAL)"
    )
    rows = []
    clean_days = ["2024-01-02", "2024-01-03"]
    for date in clean_days:
        rows.append(("^TWII", date))
        rows.extend((f"S{i}", date) for i in range(20))
    # typhoon closure: provider emitted stock rows but the benchmark is absent
    rows.extend((f"S{i}", "2024-01-04") for i in range(20))
    # dirty benchmark row on a non-trading day with almost no stocks
    rows.append(("^TWII", "2024-01-06"))
    rows.append(("S0", "2024-01-06"))
    conn.executemany(
        "INSERT INTO daily_prices VALUES (?, ?, 1, 1, 1, 1, 1, 1)", rows
    )
    conn.commit()
    conn.close()
    return db_path


def test_twse_trading_days_intersects_benchmark_and_symbol_count(calendar_db):
    days = twse_trading_days(calendar_db, min_symbols=10)
    assert days == {"2024-01-02", "2024-01-03"}


def _scored_frame(n_symbols=30, dates=("2024-01-02", "2024-04-15")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for date in dates:
        labels = rng.normal(size=n_symbols)
        for s in range(n_symbols):
            rows.append(
                {"date": date, "symbol": f"S{s}", "label": labels[s], "score": labels[s]}
            )
    return pd.DataFrame(rows)


def test_per_day_metrics_perfect_score():
    df = _scored_frame()
    daily = per_day_metrics(df, score_col="score", top_k=5)
    assert len(daily) == 2
    assert np.allclose(daily["rank_ic"], 1.0)
    assert np.allclose(daily["overlap_topk"], 1.0)
    assert (daily["top_excess"] > 0).all()


def test_per_day_metrics_inverted_score():
    df = _scored_frame()
    df["score"] = -df["score"]
    daily = per_day_metrics(df, score_col="score", top_k=5)
    assert np.allclose(daily["rank_ic"], -1.0)
    assert (daily["top_excess"] < 0).all()


def test_batch_to_matrix_respects_requested_column_order(tmp_path):
    import pyarrow.parquet as pq
    from finetune_tw.round6_diagnostics import batch_to_matrix

    # file column order (emb_1 before emb_0) differs from the requested order
    df = pd.DataFrame({
        "date": ["2024-01-02"] * 4,
        "emb_1": [10.0, 11.0, 12.0, 13.0],
        "emb_0": [0.0, 1.0, 2.0, 3.0],
        "label": [0.1, 0.2, 0.3, 0.4],
    })
    path = tmp_path / "t.parquet"
    df.to_parquet(path, index=False)
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=10))

    x = batch_to_matrix(batch, ["emb_0", "emb_1"])
    np.testing.assert_allclose(x, df[["emb_0", "emb_1"]].to_numpy(dtype=np.float32))

    x_sub = batch_to_matrix(batch, ["emb_0", "emb_1"], row_idx=np.array([1, 3]))
    np.testing.assert_allclose(x_sub, [[1.0, 11.0], [3.0, 13.0]])


def test_aggregate_period_quarterly_groups():
    df = _scored_frame(dates=("2024-01-02", "2024-01-03", "2024-04-15"))
    daily = per_day_metrics(df, score_col="score", top_k=5)
    q = aggregate_period(daily, freq="Q")
    assert list(q["period"]) == ["2024Q1", "2024Q2"]
    assert list(q["days"]) == [2, 1]
    assert np.allclose(q["mean_ic"], 1.0)


def test_stream_scores_projects_raw_columns_only(monkeypatch):
    requested_columns = []

    class FakeBooster:
        def inplace_predict(self, x, iteration_range):
            return np.zeros(len(x), dtype=np.float32)

    class FakeBatch:
        def __init__(self):
            self._table = pa.table({
                "date": ["2024-01-02", "2024-01-03"],
                "symbol": ["S0", "S1"],
                "label": [0.1, 0.2],
                "feat_ma5_dist": [1.0, 2.0],
                "feat_volume_ratio": [3.0, 4.0],
            })
            self.schema = self._table.schema

        @property
        def num_rows(self):
            return self._table.num_rows

        def select(self, columns):
            return self._table.select(columns)

        def column(self, idx):
            return self._table.column(idx).chunk(0)

    class FakeParquetFile:
        schema_arrow = pa.schema([
            ("date", pa.string()),
            ("symbol", pa.string()),
            ("label", pa.float64()),
            ("emb_0", pa.float64()),
            ("emb_1", pa.float64()),
            ("feat_ma5_dist", pa.float64()),
            ("feat_volume_ratio", pa.float64()),
        ])

        def __init__(self, path):
            self.path = path

        def iter_batches(self, batch_size, columns=None):
            requested_columns.append(columns)
            yield FakeBatch()

    monkeypatch.setattr("finetune_tw.round6_diagnostics.pq.ParquetFile", FakeParquetFile)

    scored = list(stream_scores(
        "ignored.parquet",
        FakeBooster(),
        trading_days={"2024-01-02", "2024-01-03"},
        iteration_ranges={"score_best": (0, 1)},
        feat_cols=["feat_ma5_dist", "feat_volume_ratio"],
    ))

    assert requested_columns == [[
        "date",
        "symbol",
        "label",
        "feat_ma5_dist",
        "feat_volume_ratio",
    ]]
    assert len(scored) == 1
    assert list(scored[0].columns) == [
        "date",
        "symbol",
        "label",
        "feat_ma5_dist",
        "feat_volume_ratio",
        "score_best",
    ]


def test_stream_scores_excludes_embeddings_from_pandas_output(monkeypatch):
    requested_columns = []
    selected_columns = []

    class FakeBooster:
        def inplace_predict(self, x, iteration_range):
            assert x.shape == (2, 3)
            return np.array([0.3, 0.4], dtype=np.float32)

    class FakeBatch:
        def __init__(self):
            self._table = pa.table({
                "date": ["2024-01-02", "2024-01-02"],
                "symbol": ["S0", "S1"],
                "label": [0.1, 0.2],
                "emb_0": [10.0, 11.0],
                "emb_1": [12.0, 13.0],
                "feat_ma5_dist": [1.0, 2.0],
            })
            self.schema = self._table.schema

        @property
        def num_rows(self):
            return self._table.num_rows

        def select(self, columns):
            selected_columns.append(columns)
            return self._table.select(columns)

        def column(self, idx):
            return self._table.column(idx).chunk(0)

    class FakeParquetFile:
        schema_arrow = pa.schema([
            ("date", pa.string()),
            ("symbol", pa.string()),
            ("label", pa.float64()),
            ("emb_0", pa.float64()),
            ("emb_1", pa.float64()),
            ("feat_ma5_dist", pa.float64()),
        ])

        def __init__(self, path):
            self.path = path

        def iter_batches(self, batch_size, columns=None):
            requested_columns.append(columns)
            yield FakeBatch()

    monkeypatch.setattr("finetune_tw.round6_diagnostics.pq.ParquetFile", FakeParquetFile)

    scored = list(stream_scores(
        "ignored.parquet",
        FakeBooster(),
        trading_days={"2024-01-02"},
        iteration_ranges={"score_full": (0, 0)},
        feat_cols=["emb_0", "emb_1", "feat_ma5_dist"],
    ))

    assert requested_columns == [[
        "date",
        "symbol",
        "label",
        "feat_ma5_dist",
        "emb_0",
        "emb_1",
    ]]
    assert selected_columns == [[
        "date",
        "symbol",
        "label",
        "feat_ma5_dist",
    ]]
    assert len(scored) == 1
    assert list(scored[0].columns) == [
        "date",
        "symbol",
        "label",
        "feat_ma5_dist",
        "score_full",
    ]
    assert "emb_0" not in scored[0].columns
    assert "emb_1" not in scored[0].columns


def test_iter_scored_dates_yields_only_completed_dates():
    batches = [
        pd.DataFrame({
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "symbol": ["S0", "S1", "S0"],
            "label": [0.9, 0.8, 0.1],
            "score": [0.9, 0.8, 0.1],
        }),
        pd.DataFrame({
            "date": ["2024-01-03", "2024-01-03", "2024-01-04"],
            "symbol": ["S1", "S2", "S0"],
            "label": [0.2, 0.3, 0.4],
            "score": [0.2, 0.3, 0.4],
        }),
        pd.DataFrame({
            "date": ["2024-01-04", "2024-01-04"],
            "symbol": ["S1", "S2"],
            "label": [0.5, 0.6],
            "score": [0.5, 0.6],
        }),
    ]

    streamed = list(iter_scored_dates(iter(batches)))

    assert [frame["date"].iloc[0] for frame in streamed] == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert [len(frame) for frame in streamed] == [2, 3, 3]

    metrics = [per_day_metrics(frame, score_col="score", top_k=1) for frame in streamed]
    assert [list(day["date"]) for day in metrics] == [
        ["2024-01-02"],
        ["2024-01-03"],
        ["2024-01-04"],
    ]


def test_iter_scored_dates_raises_when_a_date_reappears_in_a_later_run():
    batches = [
        pd.DataFrame({
            "date": ["2024-01-02", "2024-01-04"],
            "symbol": ["S0", "S1"],
            "label": [0.1, 0.2],
            "score": [0.1, 0.2],
        }),
        pd.DataFrame({
            "date": ["2024-01-02"],
            "symbol": ["S2"],
            "label": [0.3],
            "score": [0.3],
        }),
    ]

    with pytest.raises(ValueError, match="contiguous run"):
        list(iter_scored_dates(iter(batches)))
