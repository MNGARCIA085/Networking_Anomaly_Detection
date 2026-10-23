import numpy as np
import pytest

from network_anomaly_detection.thresholding.thresholding import Thresholding
from network_anomaly_detection.thresholding.strategies import (
    ConstrainedF1Threshold,
)
from network_anomaly_detection.thresholding.registry import (
    create_threshold_strategy,
)


def test_constrained_f1_selects_threshold_meeting_recall_constraint():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1, 1])

    strategy = ConstrainedF1Threshold(min_recall=0.66)
    strategy.fit(scores, y_true)

    predictions = scores >= strategy.get_threshold()

    recall = predictions[y_true == 1].mean()

    assert recall >= 0.66
    assert strategy.get_threshold() in np.unique(scores)


def test_constrained_f1_selects_best_f1_among_feasible_thresholds():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1, 1])

    strategy = ConstrainedF1Threshold(min_recall=1.0)
    strategy.fit(scores, y_true)

    # To achieve recall=1.0, the threshold cannot exceed
    # the lowest anomaly score, 0.3.
    assert strategy.get_threshold() == pytest.approx(0.3)


def test_fit_requires_scores():
    strategy = ConstrainedF1Threshold()

    with pytest.raises(ValueError, match="requires scores and labels"):
        strategy.fit(val_scores=None, y_val=np.array([0, 1]))


def test_fit_requires_matching_lengths():
    strategy = ConstrainedF1Threshold()

    with pytest.raises(ValueError, match="same length"):
        strategy.fit(
            val_scores=np.array([0.1, 0.2]),
            y_val=np.array([0, 1, 0]),
        )


def test_fit_fails_when_recall_constraint_is_impossible():
    strategy = ConstrainedF1Threshold(min_recall=1.1)

    with pytest.raises(ValueError, match="minimum recall constraint"):
        strategy.fit(
            val_scores=np.array([0.1, 0.2, 0.3]),
            y_val=np.array([0, 1, 1]),
        )


def test_get_threshold_requires_fit():
    strategy = ConstrainedF1Threshold()

    with pytest.raises(RuntimeError, match="not been fitted"):
        strategy.get_threshold()


def test_thresholding_disabled_returns_none():
    thresholding = Thresholding(config=None)

    thresholding.fit(
        scores=np.array([0.1, 0.2]),
        y_val=np.array([0, 1]),
    )

    assert thresholding.get_threshold() is None


def test_thresholding_wrapper_fits_strategy():
    thresholding = Thresholding(
        config={
            "name": "constrained_f1",
            "params": {"min_recall": 1.0},
        }
    )

    scores = np.array([0.1, 0.2, 0.3, 0.8])
    y_true = np.array([0, 0, 1, 1])

    result = thresholding.fit(scores, y_true)

    assert result is thresholding
    assert thresholding.get_threshold() == pytest.approx(0.3)


def test_create_threshold_strategy():
    strategy = create_threshold_strategy(
        "constrained_f1",
        min_recall=0.8,
    )

    assert isinstance(strategy, ConstrainedF1Threshold)
    assert strategy.min_recall == pytest.approx(0.8)


def test_thresholding_save_and_load(tmp_path):
    thresholding = Thresholding(
        config={
            "name": "constrained_f1",
            "params": {"min_recall": 1.0},
        }
    )

    thresholding.fit(
        scores=np.array([0.1, 0.2, 0.3, 0.8]),
        y_val=np.array([0, 0, 1, 1]),
    )

    path = tmp_path / "thresholding.pkl"
    thresholding.save(path)

    loaded = Thresholding.load(path)

    assert loaded.get_threshold() == pytest.approx(
        thresholding.get_threshold()
    )