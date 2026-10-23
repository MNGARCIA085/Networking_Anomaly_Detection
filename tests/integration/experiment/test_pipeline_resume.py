from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import network_anomaly_detection.experiment.pipeline as pipeline_module
from network_anomaly_detection.experiment.pipeline import ExperimentPipeline


class FakePrep:
    @classmethod
    def load(cls, path):
        instance = cls()
        instance.loaded_path = path
        return instance

    def select_training_data(self, X, y):
        mask = y == 0
        return X[mask], y[mask]

    def transform_with_labels(self, X, y):
        return X, y


class FakeModel:
    def adapt_input(self, X):
        return X

    def score(self, X):
        return X[:, 0]

    def predict(self, X, threshold=None):
        if threshold is None:
            return np.zeros(len(X), dtype=int)
        return (self.score(X) >= threshold).astype(int)


class FakeTrainer:
    def __init__(self):
        self.fit_args = None
        self.history = {"train_loss": [0.5], "val_loss": [0.4]}

    def fit(self, X_train, y_train, X_val, y_val, ckpt_path=None):
        self.fit_args = (X_train, y_train, X_val, y_val, ckpt_path)


@pytest.fixture
def setup_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module.BasePrep, "load", FakePrep.load)

    checkpoint = tmp_path / "epoch=1.ckpt"
    checkpoint.touch()

    logger = Mock()
    logger.get_run_artifact_path.return_value = str(tmp_path / "prep.pkl")
    logger.get_checkpoint_path.return_value = str(checkpoint)
    logger.start_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="resumed-run-123")
    )
    logger.checkpoint_dir.return_value = str(tmp_path / "new_checkpoints")

    model = FakeModel()
    trainer = FakeTrainer()

    monkeypatch.setattr(
        pipeline_module,
        "build_model",
        lambda cfg, input_shape: model,
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_trainer",
        lambda model, cfg, checkpoint_dir: trainer,
    )

    pipeline = ExperimentPipeline(
        cfg={
            "random_state": 42,
            "model_type": {
                "name": "fake_model",
                "models": {},
                "training": {},
                "thresholding": None,
            },
        },
        logger=logger,
    )

    evaluator = Mock()
    evaluator.evaluate.return_value = {"pr_auc": 0.85}

    X_train = np.array([[0.1], [0.2], [0.3]])
    y_train = np.array([0, 1, 0])
    X_val = np.array([[0.1], [0.8], [0.9]])
    y_val = np.array([0, 1, 1])

    return (
        pipeline,
        logger,
        evaluator,
        trainer,
        checkpoint,
        X_train,
        y_train,
        X_val,
        y_val,
    )


def test_resume_restores_prep_and_trains_from_checkpoint(setup_resume):
    (
        pipeline,
        logger,
        evaluator,
        trainer,
        checkpoint,
        X_train,
        y_train,
        X_val,
        y_val,
    ) = setup_resume

    result = pipeline.resume(
        run_id="original-run-123",
        evaluator=evaluator,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        run_name="resume-test",
    )

    logger.get_run_artifact_path.assert_called_once_with(
        "original-run-123",
        "preprocessing/prep.pkl",
    )
    logger.get_checkpoint_path.assert_called_once_with(
        "original-run-123",
        None,
    )
    logger.start_run.assert_called_once_with(run_name="resume-test")

    X_train_fit, y_train_fit, X_val_fit, y_val_fit, ckpt_path = (
        trainer.fit_args
    )

    # Training data is filtered to normal samples.
    np.testing.assert_array_equal(X_train_fit, [[0.1], [0.3]])
    np.testing.assert_array_equal(y_train_fit, [0, 0])

    # Callback validation data contains normal samples only.
    np.testing.assert_array_equal(X_val_fit, [[0.1]])
    np.testing.assert_array_equal(y_val_fit, [0])

    assert ckpt_path == str(checkpoint)

    # Evaluation still receives the full validation set.
    _, kwargs = evaluator.evaluate.call_args
    np.testing.assert_array_equal(kwargs["y_true"], y_val)
    np.testing.assert_array_equal(kwargs["scores"], X_val[:, 0])

    assert result["metrics"] == {"pr_auc": 0.85}
    assert result["model"] is not None
    assert result["trainer"] is trainer
    assert result["prep"] is not None

    logger.log_metrics.assert_called_once_with({"pr_auc": 0.85})
    logger.log_training_history.assert_called_once_with(trainer.history)
    logger.log_tags.assert_any_call({"status": "finished"})


@pytest.mark.parametrize(
    ("checkpoint_name", "expected_error", "message"),
    [
        ("missing.ckpt", FileNotFoundError, "Checkpoint not found"),
        ("checkpoint.txt", ValueError, "Expected a Lightning .ckpt file"),
    ],
)
def test_resume_rejects_invalid_checkpoint(
    setup_resume,
    tmp_path,
    checkpoint_name,
    expected_error,
    message,
):
    (
        pipeline,
        logger,
        evaluator,
        trainer,
        checkpoint,
        X_train,
        y_train,
        X_val,
        y_val,
    ) = setup_resume

    invalid_checkpoint = tmp_path / checkpoint_name
    if checkpoint_name == "checkpoint.txt":
        invalid_checkpoint.touch()

    logger.get_checkpoint_path.return_value = str(invalid_checkpoint)

    with pytest.raises(expected_error, match=message):
        pipeline.resume(
            run_id="original-run-123",
            evaluator=evaluator,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

    # Validation happens before opening the new MLflow run.
    logger.start_run.assert_not_called()
    logger.log_tags.assert_called_once()
    assert logger.log_tags.call_args.args[0]["status"] == "failed"


def test_resume_marks_run_failed_if_training_raises(setup_resume):
    (
        pipeline,
        logger,
        evaluator,
        trainer,
        checkpoint,
        X_train,
        y_train,
        X_val,
        y_val,
    ) = setup_resume

    trainer.fit = Mock(side_effect=RuntimeError("resume training failed"))

    with pytest.raises(RuntimeError, match="resume training failed"):
        pipeline.resume(
            run_id="original-run-123",
            evaluator=evaluator,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

    logger.log_tags.assert_any_call({
        "status": "failed",
        "error": "resume training failed",
    })