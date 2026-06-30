from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_FEATURE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def build_s1_oracle(
    tokenizer,
    db_path,
    start,
    end,
    lookback: int,
    predict_window: int,
    horizon: int,
    clip: float,
    seed: int,
    min_count: int,
) -> torch.Tensor:
    """Build an S1 oracle table once at training start.

    The public signature matches the training-plan brief. For unit tests and
    lightweight callers, `db_path` may also be an iterable of raw samples
    instead of a filesystem path. Raw samples should provide `s1_ids` and
    `open_prices`.
    """
    _validate_oracle_args(lookback=lookback, horizon=horizon, min_count=min_count)
    _ = seed

    if isinstance(db_path, (str, PathLike)):
        samples = _iter_db_samples(
            tokenizer=tokenizer,
            db_path=str(db_path),
            start=start,
            end=end,
            lookback=lookback,
            predict_window=predict_window,
            clip=clip,
        )
    else:
        samples = db_path

    return build_s1_oracle_from_samples(
        samples=samples,
        horizon=horizon,
        min_count=min_count,
        vocab_size=2 ** int(tokenizer.s1_bits),
    )


def build_s1_oracle_from_samples(samples, horizon, min_count, vocab_size=1024) -> torch.Tensor:
    """Map pre-sliced S1 context ids to their mean open-to-open return at `horizon`."""
    _validate_oracle_sample_args(horizon=horizon, min_count=min_count, vocab_size=vocab_size)

    sums = torch.zeros(vocab_size, dtype=torch.float64)
    counts = torch.zeros(vocab_size, dtype=torch.int64)

    for sample in samples:
        parsed = _extract_sample_fields(sample=sample)
        if parsed is None:
            continue

        last_s1, open_prices, lookback = parsed
        if last_s1 < 0 or last_s1 >= vocab_size:
            continue

        realized_return = _compute_open_to_open_return(
            open_prices=open_prices,
            lookback=lookback,
            horizon=horizon,
        )
        if realized_return is None:
            continue

        sums[last_s1] += realized_return
        counts[last_s1] += 1

    oracle = torch.zeros(vocab_size, dtype=torch.float32)
    eligible = counts >= min_count
    if eligible.any():
        oracle[eligible] = (sums[eligible] / counts[eligible].to(torch.float64)).to(torch.float32)
    return oracle


def oracle_pred_score(
    s1_logits_at_h: torch.Tensor,
    oracle: torch.Tensor,
) -> torch.Tensor:
    """Return differentiable oracle-weighted scores for a batch of logits."""
    oracle_vec = torch.as_tensor(
        oracle,
        device=s1_logits_at_h.device,
        dtype=s1_logits_at_h.dtype,
    )
    probs = F.softmax(s1_logits_at_h, dim=-1)
    return probs @ oracle_vec


def _validate_oracle_args(lookback: int, horizon: int, min_count: int) -> None:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if min_count < 0:
        raise ValueError("min_count must be non-negative")


def _validate_oracle_sample_args(horizon: int, min_count: int, vocab_size: int) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if min_count < 0:
        raise ValueError("min_count must be non-negative")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")


def _iter_db_samples(
    tokenizer,
    db_path: str,
    start,
    end,
    lookback: int,
    predict_window: int,
    clip: float,
):
    from finetune_tw.db import list_symbols, query_symbol

    total_window = lookback + predict_window + 1
    for symbol in list_symbols(db_path):
        frame = query_symbol(db_path, symbol, start=start, end=end)
        if len(frame) < total_window:
            continue

        values = frame.loc[:, _FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        for start_idx in range(len(values) - total_window + 1):
            raw_window = values[start_idx : start_idx + total_window]
            normalized_window = _normalize_window(raw_window, lookback=lookback, clip=clip)
            s1_ids = _encode_s1_sequence(tokenizer, torch.from_numpy(normalized_window))
            yield {
                "s1_ids": s1_ids[:lookback].cpu(),
                "open_prices": raw_window[:, 0].copy(),
            }


def _normalize_window(window: np.ndarray, lookback: int, clip: float) -> np.ndarray:
    past = window[:lookback]
    mean = past.mean(axis=0)
    std = past.std(axis=0) + 1e-5
    normalized = (window - mean) / std
    return np.clip(normalized, -clip, clip).astype(np.float32, copy=False)


def _extract_sample_fields(sample: Any) -> tuple[int, Any, int] | None:
    if isinstance(sample, dict):
        return _extract_from_mapping(sample=sample)
    return None


def _extract_from_mapping(sample: dict[str, Any]) -> tuple[int, Any, int] | None:
    open_prices = _mapping_open_prices(sample)
    if open_prices is None:
        return None

    if "s1_ids" in sample:
        s1_ids = _as_1d_long_tensor(sample["s1_ids"])
        if s1_ids.numel() < 1:
            return None
        return int(s1_ids[-1].item()), open_prices, int(s1_ids.numel())

    return None


def _mapping_open_prices(sample: dict[str, Any]) -> Any | None:
    if "open_prices" in sample:
        return sample["open_prices"]
    if "opens" in sample:
        return sample["opens"]
    if "x" in sample:
        x = _as_feature_tensor(sample["x"])
        if x is None:
            return None
        return x[:, 0]
    return None


def _extract_from_tuple(sample: tuple[Any, ...] | list[Any], tokenizer, lookback: int) -> tuple[int, Any] | None:
    return _extract_from_x(sample[0], tokenizer=tokenizer, lookback=lookback)


def _extract_from_x(x: Any, tokenizer, lookback: int) -> tuple[int, Any] | None:
    features = _as_feature_tensor(x)
    if features is None or features.shape[0] < lookback:
        return None

    s1_ids = _encode_s1_sequence(tokenizer, features)
    if s1_ids.numel() < lookback:
        return None

    last_s1 = int(s1_ids[lookback - 1].item())
    return last_s1, features[:, 0]


def _as_feature_tensor(value: Any) -> torch.Tensor | None:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[1] < 1:
        return None
    return tensor


def _as_1d_long_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    return tensor.reshape(-1)


def _compute_open_to_open_return(open_prices: Any, lookback: int, horizon: int) -> float | None:
    opens = torch.as_tensor(open_prices, dtype=torch.float32).reshape(-1)
    target_idx = lookback + horizon
    if opens.numel() <= target_idx:
        return None

    base = float(opens[lookback].item())
    future = float(opens[target_idx].item())
    if not np.isfinite(base) or not np.isfinite(future) or base <= 0.0:
        return None

    realized_return = future / base - 1.0
    if not np.isfinite(realized_return):
        return None
    return realized_return


def _encode_s1_sequence(tokenizer, x_window: torch.Tensor) -> torch.Tensor:
    batch = x_window.unsqueeze(0).to(_tokenizer_device(tokenizer))
    try:
        encoded = tokenizer.encode(batch, half=True)
    except TypeError:
        encoded = tokenizer.encode(batch)

    s1_ids = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
    s1_tensor = torch.as_tensor(s1_ids, dtype=torch.long)
    if s1_tensor.ndim == 2 and s1_tensor.shape[0] == 1:
        s1_tensor = s1_tensor[0]
    return s1_tensor.reshape(-1).cpu()


def _tokenizer_device(tokenizer) -> torch.device:
    parameters = getattr(tokenizer, "parameters", None)
    if callable(parameters):
        try:
            first_param = next(iter(parameters()))
        except Exception:
            first_param = None
        if first_param is not None and hasattr(first_param, "device"):
            return first_param.device
    return torch.device("cpu")
