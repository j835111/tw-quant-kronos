from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardFold:
    train_start: str
    train_end: str
    embargo_end: str
    val_start: str
    val_end: str


def single_fold(
    train_start: str,
    train_end: str,
    val_end: str,
    embargo_days: int = 110,
) -> WalkForwardFold:
    embargo_end = (
        pd.Timestamp(train_end) + pd.Timedelta(days=embargo_days)
    ).strftime("%Y-%m-%d")
    return WalkForwardFold(
        train_start=train_start,
        train_end=train_end,
        embargo_end=embargo_end,
        val_start=embargo_end,
        val_end=val_end,
    )


def oof_folds(
    start: str,
    end: str,
    n_folds: int = 5,
    embargo_days: int = 110,
) -> list[WalkForwardFold]:
    bdays = pd.bdate_range(start, end)
    if len(bdays) < (n_folds + 1) * 2:
        raise ValueError(
            f"Not enough business days ({len(bdays)}) for {n_folds} folds "
            f"(need at least {(n_folds + 1) * 2})"
        )

    fold_size = len(bdays) // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end = bdays[fold_size * (i + 1) - 1]
        val_end = bdays[min(fold_size * (i + 2) - 1, len(bdays) - 1)]
        folds.append(
            single_fold(
                train_start=start,
                train_end=train_end.strftime("%Y-%m-%d"),
                val_end=val_end.strftime("%Y-%m-%d"),
                embargo_days=embargo_days,
            )
        )
    return folds
