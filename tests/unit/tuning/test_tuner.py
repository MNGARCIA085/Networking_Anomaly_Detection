from unittest.mock import Mock

import numpy as np
import optuna
import pytest

from network_anomaly_detection.tuning.tuner import Tuner


@pytest.fixture
def cfg():
    return {
        "random_state": 42,
        "paths": {"data": "data/processed"},
        "tracking": {"uri": "mlflow.db"},
        "experiment_name": "anomaly-detection",
        "model_type": {
            "name": "ae",
            "tuning": {
                "n_trials": 2,
                "n_jobs": 1,
                "metric": "pr_auc",
                "direction": "maximize",
            },
        },
    }


@pytest.fixture
def data():
    X_train = np.array([[0.1], [0.2], [0.3]])
    y_train = np.array([0, 0, 1])

    X_val = np.array([[0.4], [0.5]])
    y_val = np.array([0, 1])

    return X_train, y_train, X_val, y_val


@pytest.fixture
def tuner(cfg, monkeypatch):
    import network_anomaly_detection.tuning.tuner as tuner_module

    sampler = Mock()
    sampler.sample.side_effect = [
        {
            "model_type": {
                "name": "ae",
                "models": {"encoder_dims": [32, 16]},
                "tuning": {"n_trials": 2},
            }
        },
        {
            "model_type": {
                "name": "ae",
                "models": {"encoder_dims": [64, 32]},
                "tuning": {"n_trials": 2},
            }
        },
    ]

    monkeypatch.setitem(
        tuner_module.TUNING_REGISTRY,
        "ae",
        lambda base_cfg, tuning_cfg: sampler,
    )

    experiment = Mock()
    experiment.run.return_value = {
        "metrics": {"pr_auc": 0.8}
    }

    monkeypatch.setattr(
        tuner_module,
        "Experiment",
        lambda cfg, logger: experiment,
    )

    prep = Mock()
    prep.select_training_data.side_effect = lambda X, y: (X, y)
    prep_cls = Mock(return_value=prep)

    monkeypatch.setitem(
        tuner_module.PREP_REGISTRY,
        "ae",
        prep_cls,
    )

    instance = Tuner(
        cfg=cfg,
        evaluator=Mock(),
        logger=Mock(),
    )

    return instance, sampler, experiment, prep, prep_cls


def test_run_optimizes_configured_metric(tuner, data):
    instance, sampler, experiment, prep, prep_cls = tuner
    X_train, y_train, X_val, y_val = data

    study = instance.run(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        parent_run_id="parent-123",
    )

    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert len(study.trials) == 2

    assert all(
        trial.value == pytest.approx(0.8)
        for trial in study.trials
    )

    assert sampler.sample.call_count == 2
    assert experiment.run.call_count == 2
    assert prep_cls.call_count == 2
    assert prep.fit.call_count == 2

    # Each trial records the configuration it sampled.
    for trial in study.trials:
        assert trial.user_attrs["config"]["model_type"]["name"] == "ae"


def test_trial_uses_sampled_config_and_parent_metadata(tuner, data):
    instance, sampler, experiment, prep, prep_cls = tuner
    X_train, y_train, X_val, y_val = data

    instance.run(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        parent_run_id="parent-123",
        n_trials=1,
        n_jobs=1,
    )

    call = experiment.run.call_args

    assert call.kwargs["nested"] is True
    assert call.kwargs["run_name"] == "trial_0"

    assert call.kwargs["metadata"] == {
        "mlflow.parentRunId": "parent-123",
        "run_type": "tuning_trial",
        "trial_number": "0",
        "tuning_metric": "pr_auc",
    }

    assert call.kwargs["prep"] is prep
    assert call.kwargs["evaluator"] is instance.evaluator

    # The sampled training data and labels are passed to the experiment.
    np.testing.assert_array_equal(
        call.kwargs["X_train"],
        X_train,
    )
    np.testing.assert_array_equal(
        call.kwargs["y_train"],
        y_train,
    )

    # Validation data is not filtered by Tuner.
    np.testing.assert_array_equal(
        call.kwargs["X_val"],
        X_val,
    )
    np.testing.assert_array_equal(
        call.kwargs["y_val"],
        y_val,
    )


def test_get_best_config_removes_tuning_and_tracking_keys():
    best_config = {
        "random_state": 42,
        "paths": {"data": "data/processed"},
        "tracking": {"uri": "mlflow.db"},
        "experiment_name": "anomaly-detection",
        "model_type": {
            "name": "ae",
            "models": {"encoder_dims": [64, 32]},
            "tuning": {"n_trials": 20},
        },
    }

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial.set_user_attr("config", best_config)
    study.tell(trial, 0.95)

    result = Tuner.get_best_config(study)

    assert result["model_type"]["models"]["encoder_dims"] == [64, 32]

    assert "tuning" not in result["model_type"]
    assert "paths" not in result
    assert "tracking" not in result
    assert "experiment_name" not in result

    # Extraction must not mutate the configuration stored on the trial.
    assert (
        "tuning"
        in study.best_trial.user_attrs["config"]["model_type"]
    )