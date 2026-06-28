import pytest
import torch

from finetune_tw.train_predictor import (
    _combine_training_loss,
    differentiable_rank_ic_loss,
)


def test_differentiable_rank_ic_loss_perfect():
    pred = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    actual = pred.clone()

    loss = differentiable_rank_ic_loss(pred, actual)

    assert loss.item() == pytest.approx(-1.0, abs=1e-6)


def test_differentiable_rank_ic_loss_reversed():
    actual = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    pred = -actual

    loss = differentiable_rank_ic_loss(pred, actual)

    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_differentiable_rank_ic_loss_single_item():
    pred = torch.tensor([0.1], dtype=torch.float32)
    actual = torch.tensor([0.2], dtype=torch.float32)

    loss = differentiable_rank_ic_loss(pred, actual)

    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_ranking_loss_alpha_zero_no_effect():
    token_loss = torch.tensor(1.2345, dtype=torch.float32)
    ranking_loss = torch.tensor(-0.75, dtype=torch.float32)

    total_loss = _combine_training_loss(token_loss, ranking_loss_alpha=0.0, ranking_loss=ranking_loss)

    assert total_loss.item() == pytest.approx(token_loss.item(), abs=1e-9)
