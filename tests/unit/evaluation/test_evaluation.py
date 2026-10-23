import numpy as np
import pytest

from network_anomaly_detection.evaluation.evaluator import Evaluator


@pytest.fixture
def evaluator():
    return Evaluator()


def test_evaluate_without_labels_returns_mean_score(evaluator):
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = evaluator.evaluate(scores)

    assert result == {"mean_score": pytest.approx(0.5)}


def test_evaluate_with_labels_returns_threshold_independent_metrics(evaluator):
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = evaluator.evaluate(scores, y_true=y_true)

    assert result["mean_score"] == pytest.approx(0.5)
    assert result["auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)

    # No predictions: threshold-dependent metrics should be absent.
    assert "precision" not in result
    assert "recall" not in result
    assert "f1" not in result
    assert "tn" not in result


def test_evaluate_with_predictions_returns_classification_metrics(evaluator):
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    predictions = np.array([0, 1, 1, 1])

    result = evaluator.evaluate(
        scores,
        y_true=y_true,
        predictions=predictions,
    )

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)

    assert result["tn"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["tp"] == 2


def test_evaluate_without_predictions_omits_confusion_matrix(evaluator):
    y_true = np.array([0, 1])
    scores = np.array([0.1, 0.9])

    result = evaluator.evaluate(scores, y_true=y_true)

    assert not {"precision", "recall", "f1", "tn", "fp", "fn", "tp"} & result.keys()


def test_evaluate_handles_zero_precision(evaluator):
    y_true = np.array([0, 1, 1])
    scores = np.array([0.1, 0.8, 0.9])
    predictions = np.array([0, 0, 0])

    result = evaluator.evaluate(
        scores,
        y_true=y_true,
        predictions=predictions,
    )

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0

    assert result["tn"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 2
    assert result["tp"] == 0