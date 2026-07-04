from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

from finetune_tw.feature_engineering import TECH_FEATURE_COLUMNS, technical_feature_columns
from finetune_tw.ic_validation import rank_ic
from finetune_tw.lambdarank_ic import lambdarank_ic_objective

EMBEDDING_PREFIX = "emb_"
_TECH_FEATURE_COLUMNS = TECH_FEATURE_COLUMNS


def build_group_sizes(df: pd.DataFrame, date_col: str = "date") -> list[int]:
    """Row order MUST already be sorted by date_col before calling this (caller's responsibility)."""
    return df.groupby(date_col, sort=False).size().tolist()


def rank_ic_eval_metric(preds: np.ndarray, dtrain, group_sizes: list[int]) -> float:
    labels = dtrain.get_label()
    boundaries = np.cumsum([0] + list(group_sizes))
    ics = [
        rank_ic(preds[start:end], labels[start:end])
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics)) if ics else float("nan")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    emb_cols = sorted([c for c in df.columns if c.startswith(EMBEDDING_PREFIX)],
                      key=lambda c: int(c[len(EMBEDDING_PREFIX):]))
    tech_cols = technical_feature_columns(df.columns)
    return emb_cols + tech_cols


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
    params: dict | None = None,
) -> xgb.Booster:
    train_df = train_df.sort_values("date", kind="stable").reset_index(drop=True)
    val_df = val_df.sort_values("date", kind="stable").reset_index(drop=True)

    feat_cols = _feature_columns(train_df)
    train_groups = build_group_sizes(train_df)
    val_groups = build_group_sizes(val_df)

    dtrain = xgb.DMatrix(train_df[feat_cols].values, label=train_df["label"].values)
    dtrain.set_group(train_groups)
    dval = xgb.DMatrix(val_df[feat_cols].values, label=val_df["label"].values)
    dval.set_group(val_groups)

    obj = lambdarank_ic_objective(train_groups, sigma=1.0)

    def feval(preds, dmat):
        return "rank_ic", -rank_ic_eval_metric(preds, dmat, val_groups)  # XGBoost minimizes eval metric

    default_params = {"max_depth": 4, "eta": 0.05, "tree_method": "hist"}
    booster = xgb.train(
        {**default_params, **(params or {})},
        dtrain,
        num_boost_round=num_boost_round,
        obj=obj,
        evals=[(dval, "val")],
        custom_metric=feval,
        maximize=False,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10,
    )
    return booster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Parquet from extract_embeddings.py")
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True, help="Output path for the trained booster (.json)")
    parser.add_argument("--num_boost_round", type=int, default=200)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    booster = train(train_df, val_df, args.num_boost_round, args.early_stopping_rounds)
    booster.save_model(args.out)
    print(f"Saved -> {args.out}  (best_iteration={booster.best_iteration})")


if __name__ == "__main__":
    main()
