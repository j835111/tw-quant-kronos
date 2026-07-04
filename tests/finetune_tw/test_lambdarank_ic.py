import numpy as np

from finetune_tw.lambdarank_ic import lambdarank_ic_grad_hess, lambdarank_ic_objective


def test_gradient_sign_pushes_misranked_pair_toward_correct_order():
    # labels: sample 0 is best (3), sample 2 is worst (1). preds are fully reversed.
    labels = np.array([3.0, 2.0, 1.0])
    preds = np.array([1.0, 2.0, 3.0])
    grad, hess = lambdarank_ic_grad_hess(preds, labels)

    assert grad[0] < 0    # best label, lowest pred -> must be pushed up
    assert grad[2] > 0    # worst label, highest pred -> must be pushed down
    assert (hess > 0).all()


def test_gradient_is_zero_for_perfectly_ranked_group():
    labels = np.array([3.0, 2.0, 1.0])
    preds = np.array([30.0, 20.0, 10.0])  # already in perfect descending order matching labels
    grad, _ = lambdarank_ic_grad_hess(preds, labels, sigma=1.0)
    # rho ~ 0 for confidently-correct pairs -> gradient magnitude near zero
    assert np.allclose(grad, 0.0, atol=1e-3)


def test_single_sample_group_returns_zero_gradient():
    grad, hess = lambdarank_ic_grad_hess(np.array([1.0]), np.array([1.0]))
    assert grad.shape == (1,)
    assert np.allclose(grad, 0.0)
    assert np.allclose(hess, 0.0)


class _FakeDMatrix:
    def __init__(self, labels):
        self._labels = labels

    def get_label(self):
        return self._labels


def test_objective_respects_group_boundaries():
    # 2 groups: [3.0, 2.0, 1.0] (2 samples) and [1.0, 5.0] (2 samples, mixed with group 1's labels)
    labels = np.array([3.0, 1.0, 1.0, 5.0])
    preds = np.array([1.0, 2.0, 2.0, 1.0])
    obj = lambdarank_ic_objective(group_sizes=[2, 2])
    grad, hess = obj(preds, _FakeDMatrix(labels))

    assert grad.shape == (4,)
    # group 1 (indices 0,1): label 3 > label 1 -> sample 0 (pred=1, lower) should be pushed up
    assert grad[0] < 0
    # group 2 (indices 2,3): label 1 < label 5 -> sample 3 (pred=1, lower) should be pushed up
    assert grad[3] < 0
