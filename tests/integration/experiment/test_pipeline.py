import numpy as np
import pytest
from unittest.mock import Mock

from network_anomaly_detection.experiment.pipeline import ExperimentPipeline


class FakePrep:
    """Already-fitted preprocessing component."""

    def __init__(self):
        self.transform_calls = []
        self.selection_calls = []

    def transform_with_labels(self, X, y):
        self.transform_calls.append((X.copy(), y.copy()))
        return X, y

    def select_training_data(self, X, y):
        self.selection_calls.append((X.copy(), y.copy()))
        mask = y == 0
        return X[mask], y[mask]


class FakeModel:
    def __init__(self):
        self.adapted_shapes = []

    def adapt_input(self, X):
        self.adapted_shapes.append(X.shape)
        return X

    def score(self, X):
        # Deterministic scores, one per sample.
        return X[:, 0]

    def predict(self, X, threshold=None):
        if threshold is None:
            return np.zeros(len(X), dtype=int)
        return (self.score(X) >= threshold).astype(int)


class FakeTrainer:
    def __init__(self):
        self.fit_args = None
        self.history = {"train_loss": [0.5], "val_loss": [0.4]}

    def fit(self, X_train, y_train, X_val, y_val):
        self.fit_args = (X_train, y_train, X_val, y_val)


@pytest.fixture
def pipeline(monkeypatch):
    import network_anomaly_detection.experiment.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.mlflow, "log_dict", Mock())

    cfg = {
        "random_state": 42,
        "model_type": {
            "name": "fake_model",
            "models": {},
            "training": {},
            "thresholding": None,
        },
    }

    pipeline = ExperimentPipeline(cfg=cfg, logger=Mock())
    prep = FakePrep()
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

    pipeline.logger.checkpoint_dir.return_value = "/tmp/checkpoints"

    return pipeline, prep, model, trainer


def test_run_transforms_data_trains_and_evaluates_full_validation(
    pipeline,
):
    pipeline, prep, model, trainer = pipeline

    X_train = np.array([[0.1], [0.2], [0.3]])
    y_train = np.array([0, 0, 0])

    X_val = np.array([[0.1], [0.8], [0.9]])
    y_val = np.array([0, 1, 1])

    evaluator = Mock()
    evaluator.evaluate.return_value = {"pr_auc": 0.9}

    result = pipeline.run(
        prep=prep,
        evaluator=evaluator,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    # Preprocessing transforms both datasets; it is not fitted here.
    assert len(prep.transform_calls) == 2

    # The trainer receives the normal-only validation subset for callbacks.
    _, _, X_val_train, y_val_train = trainer.fit_args
    np.testing.assert_array_equal(X_val_train, [[0.1]])
    np.testing.assert_array_equal(y_val_train, [0])

    # Evaluation uses the complete validation set, including anomalies.
    evaluator.evaluate.assert_called_once()
    _, kwargs = evaluator.evaluate.call_args
    np.testing.assert_array_equal(kwargs["y_true"], y_val)
    np.testing.assert_array_equal(kwargs["scores"], X_val[:, 0])

    assert result["metrics"] == {"pr_auc": 0.9}
    assert result["model"] is model
    assert result["trainer"] is trainer

    pipeline.logger.log_metrics.assert_called_once_with({"pr_auc": 0.9})
    pipeline.logger.log_candidate_model.assert_called_once()


def test_run_logs_training_history_when_available(pipeline):
    pipeline, prep, model, trainer = pipeline

    X_train = np.array([[0.1], [0.2]])
    y_train = np.array([0, 0])
    X_val = np.array([[0.3], [0.9]])
    y_val = np.array([0, 1])

    evaluator = Mock()
    evaluator.evaluate.return_value = {"pr_auc": 0.8}

    pipeline.run(
        prep=prep,
        evaluator=evaluator,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    pipeline.logger.log_training_history.assert_called_once_with(
        trainer.history
    )




"""
The pipeline uses the already-fitted preprocessing object; it does not fit it.
The trainer gets the selected validation subset, while evaluation uses the full validation set.
Training, scoring, metric logging, and candidate-model logging are connected correctly.
Training history is logged when availabl

pytest tests/integration/experiment/test_pipeline.py -q


"""