from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from network_anomaly_detection.experiment.experiment import Experiment


@pytest.fixture
def logger():
    logger = Mock()
    logger.start_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="test-run-123")
    )
    return logger


@pytest.fixture
def experiment(logger):
    exp = Experiment(cfg={}, logger=logger)
    exp.pipeline = Mock()
    return exp


def test_start_creates_run_and_records_run_id(experiment, logger):
    run = experiment.start(run_name="test-run")

    logger.start_run.assert_called_once_with(
        run_name="test-run",
        nested=False,
        parent_run_id=None,
    )
    logger.log_tags.assert_called_once_with({"status": "running"})

    assert experiment.run_id == "test-run-123"
    assert run.info.run_id == experiment.run_id


def test_run_executes_pipeline_and_finishes(experiment, logger):
    expected_result = {"metrics": {"pr_auc": 0.85}}
    experiment.pipeline.run.return_value = expected_result

    result = experiment.run(
        prep=Mock(),
        evaluator=Mock(),
        X_train=[1, 2],
        y_train=[0, 0],
        X_val=[3, 4],
        y_val=[0, 1],
        run_name="test-run",
    )

    assert result == expected_result
    experiment.pipeline.run.assert_called_once()

    logger.log_tags.assert_any_call({"status": "running"})
    logger.log_tags.assert_any_call({"status": "finished"})
    logger.end_run.assert_called_once()


def test_run_marks_failure_and_reraises_exception(experiment, logger):
    error = RuntimeError("training failed")
    experiment.pipeline.run.side_effect = error

    with pytest.raises(RuntimeError, match="training failed"):
        experiment.run(
            prep=Mock(),
            evaluator=Mock(),
            X_train=[1, 2],
            y_train=[0, 0],
            X_val=[3, 4],
            y_val=[0, 1],
        )

    logger.log_tags.assert_any_call({"status": "running"})
    logger.log_tags.assert_any_call({
        "status": "failed",
        "error": "training failed",
    })
    logger.end_run.assert_called_once()