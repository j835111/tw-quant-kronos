"""Batch-3 CPU-only parquet enrichment for cached Round-6 embeddings artifacts.

Recomputes the expanded technical feature set from tw_stocks.db, drops stale feat_* columns
from the cached parquet, appends the new features plus per-date cs-ranks, and optionally
filters out phantom pd.bdate_range days using the real TWSE trading calendar.

Intended for the rechunked ext4 copies used by Batch 2/3 (`/home/james/round6_artifacts`):
the original 11GB train parquet's oversized row groups can still spike memory when decoded.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finetune_tw.db import query_symbols_window
from finetune_tw.feature_engineering import (
    TECH_FEATURE_COLUMNS,
    compute_ranked_feature_block,
    technical_feature_columns,
)
from finetune_tw.round6_diagnostics import _date_strings
from finetune_tw.trading_calendar import twse_trading_days

DEFAULT_BATCH_SIZE = 50_000
DEFAULT_BUFFER_DAYS = 180


def _iter_date_blocks(
    parquet_path: str,
    keep_dates: set[str] | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[tuple[str, pd.DataFrame]]:
    pf = pq.ParquetFile(parquet_path)
    pending: list[pd.DataFrame] = []
    current_date: str | None = None
    seen_dates: set[str] = set()
    for batch in pf.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        chunk["date"] = _date_strings(chunk["date"])
        if keep_dates is not None:
            chunk = chunk.loc[chunk["date"].isin(keep_dates)].copy()
        if chunk.empty:
            continue
        dates = chunk["date"].to_numpy()
        start = 0
        while start < len(chunk):
            date = dates[start]
            end = start + 1
            while end < len(chunk) and dates[end] == date:
                end += 1
            group = chunk.iloc[start:end].copy()
            if current_date is None:
                current_date = date
            if date != current_date:
                seen_dates.add(current_date)
                if date in seen_dates:
                    raise ValueError(
                        "parquet dates must appear in one contiguous run after filtering; "
                        f"saw {date} in multiple runs"
                    )
                yield current_date, pd.concat(pending, ignore_index=True)
                pending = [group]
                current_date = date
            else:
                pending.append(group)
            start = end
    if pending and current_date is not None:
        yield current_date, pd.concat(pending, ignore_index=True)


def _feature_block_for_dates(
    db_path: str,
    block: pd.DataFrame,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> pd.DataFrame:
    if block.empty:
        columns = ["date", "symbol", *TECH_FEATURE_COLUMNS]
        columns += [f"{col}_cs_rank" for col in TECH_FEATURE_COLUMNS]
        return pd.DataFrame(columns=columns)

    symbols = sorted(block["symbol"].unique())
    min_date = pd.Timestamp(block["date"].min()) - pd.Timedelta(days=buffer_days)
    max_date = pd.Timestamp(block["date"].max())
    history = query_symbols_window(
        db_path,
        symbols,
        start=min_date.strftime("%Y-%m-%d"),
        end=max_date.strftime("%Y-%m-%d"),
    )
    return compute_ranked_feature_block(
        history,
        block,
        feature_cols=TECH_FEATURE_COLUMNS,
        assume_sorted=True,
        strict=False,
    )


def _merge_feature_block(block: pd.DataFrame, feature_block: pd.DataFrame) -> pd.DataFrame:
    return block.merge(
        feature_block,
        on=["date", "symbol"],
        how="left",
        validate="1:1",
        sort=False,
    )


def enrich_parquet(
    input_path: str,
    output_path: str,
    db_path: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    keep_dates: set[str] | None = None,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> None:
    pf = pq.ParquetFile(input_path)
    old_feat_cols = [c for c in pf.schema_arrow.names if c.startswith("feat_")]
    original_cols = [c for c in pf.schema_arrow.names if c not in old_feat_cols]
    required_feat_cols = [
        *TECH_FEATURE_COLUMNS,
        *[f"{col}_cs_rank" for col in TECH_FEATURE_COLUMNS],
    ]
    writer = None
    try:
        for _, block in _iter_date_blocks(input_path, keep_dates=keep_dates, batch_size=batch_size):
            base = block.drop(columns=old_feat_cols, errors="ignore")
            feature_block = _feature_block_for_dates(db_path, base, buffer_days=buffer_days)
            enriched = _merge_feature_block(base, feature_block)
            missing = enriched[required_feat_cols].isna().any(axis=1)
            if missing.any():
                sample = enriched.loc[missing, ["date", "symbol"]].head(5).to_dict("records")
                print(
                    "Dropping parquet rows without same-day DB prices for feature recompute: "
                    f"{missing.sum()} rows; sample={sample}",
                    flush=True,
                )
                enriched = enriched.loc[~missing].reset_index(drop=True)
            if enriched.empty:
                continue
            ordered_cols = [c for c in original_cols if c in enriched.columns]
            ordered_cols += technical_feature_columns(enriched.columns)
            enriched = enriched.loc[:, ordered_cols]

            table = pa.Table.from_pandas(enriched, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table, row_group_size=batch_size)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--db", default="finetune_tw/data/tw_stocks.db")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--buffer-days", type=int, default=DEFAULT_BUFFER_DAYS)
    parser.add_argument("--no-twse-filter", action="store_true")
    args = parser.parse_args()

    keep_dates = None if args.no_twse_filter else twse_trading_days(args.db)
    if keep_dates is not None:
        print(f"TWSE filter on: {len(keep_dates)} trading days", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    enrich_parquet(
        args.input,
        args.output,
        args.db,
        batch_size=args.batch_size,
        keep_dates=keep_dates,
        buffer_days=args.buffer_days,
    )
    print(f"Saved enriched parquet -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
