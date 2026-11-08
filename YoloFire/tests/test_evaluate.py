import pytest

from evaluate import match_predictions, pr_at


def test_pr_at_fixed_conf():
    confs, tps = [0.9, 0.5, 0.3, 0.2], [True, False, True, True]
    assert pr_at(confs, tps, n_gt=4, conf=0.25) == pytest.approx((2 / 3, 2 / 4))
    assert pr_at(confs, tps, n_gt=4, conf=0.95) == (None, 0.0)


def test_match_predictions_greedy_by_conf():
    gts = [(0, 0, 10, 10)]
    preds = [(0.5, (0, 0, 10, 10)), (0.9, (1, 1, 10, 10))]
    confs, tp = match_predictions(preds, gts)
    # higher-confidence prediction claims the GT; the other is a false positive
    assert sorted(zip(confs, tp)) == [(0.5, False), (0.9, True)]


def test_match_predictions_no_gt():
    assert match_predictions([(0.9, (0, 0, 5, 5))], []) == ([0.9], [False])
