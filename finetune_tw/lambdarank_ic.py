"""LambdaRank-style pairwise objective targeting Spearman Rank-IC, for XGBoost custom obj=.

Standard LambdaRank pairwise-logistic gradient, with the usual |ΔNDCG| gain term replaced by
the label-rank distance |rank(y_i) - rank(y_j)| (Spearman IC's sensitivity to swapping a pair's
predicted order is linear in that rank distance). This is our derivation for Rank-IC, not a
verbatim copy of arXiv:2605.00501 Eq. 5 — reconcile against the paper later if needed.
"""
from __future__ import annotations

import numpy as np


def _dense_rank(values: np.ndarray) -> np.ndarray:
    """1-based ascending rank, ties broken by stable sort order (matches pandas .rank(method='first'))."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def lambdarank_ic_grad_hess(
    preds: np.ndarray,
    labels: np.ndarray,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient/hessian for one cross-sectional group (one trading date)."""
    n = len(preds)
    grad = np.zeros(n, dtype=np.float64)
    hess = np.zeros(n, dtype=np.float64)
    if n < 2:
        return grad, hess

    label_ranks = _dense_rank(labels)

    # i should outrank j whenever label_i > label_j; vectorized over all pairs in the group.
    pred_diff = preds[:, None] - preds[None, :]          # s_i - s_j
    label_diff = labels[:, None] - labels[None, :]       # y_i - y_j
    rank_dist = np.abs(label_ranks[:, None] - label_ranks[None, :])

    pair_mask = label_diff > 0                            # only pairs where i should outrank j
    rho = 1.0 / (1.0 + np.exp(sigma * pred_diff))          # sigmoid(-sigma * pred_diff)
    lam = sigma * rho * rank_dist * pair_mask              # magnitude, zero outside mask
    hess_pair = (sigma ** 2) * rho * (1.0 - rho) * rank_dist * pair_mask

    # i is pushed up (negative grad), j is pushed down (positive grad).
    grad += -lam.sum(axis=1) + lam.sum(axis=0)
    hess += hess_pair.sum(axis=1) + hess_pair.sum(axis=0)

    hess = np.maximum(hess, 1e-6)  # XGBoost requires strictly positive hessian
    return grad, hess


def lambdarank_ic_objective(group_sizes: list[int], sigma: float = 1.0):
    """Return an XGBoost-compatible obj(preds, dtrain) -> (grad, hess) for the whole training set."""
    boundaries = np.cumsum([0] + list(group_sizes))

    def _obj(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label()
        grad = np.zeros_like(preds, dtype=np.float64)
        hess = np.zeros_like(preds, dtype=np.float64)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            g, h = lambdarank_ic_grad_hess(preds[start:end], labels[start:end], sigma=sigma)
            grad[start:end] = g
            hess[start:end] = h
        return grad, hess

    return _obj
