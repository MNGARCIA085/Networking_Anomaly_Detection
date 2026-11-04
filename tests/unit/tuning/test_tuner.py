from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import optuna
import pytest

from network_anomaly_detection.tuning.tuner import Tuner


@pytest.fixture
def cfg():
    """Base experiment configuration."""
    return {
        "random_state": 42,
        "paths": {"data": "data/processed"},
        "tracking": {"uri": "mlflow.db"},
        "experiment_name": "anomaly-detection",
        "name": "ae",
        "models": {
            "encoder_dims": [128, 64, 32, 16],
            "decoder_dims": [32, 64, 128],
        },
        "training": {
            "batch_size": 32,
            "epochs": 5,
        },
        "prep": {
            "pointwise": {
                "imputation": {
                    "name": "simple",
                    "params": {"strategy": "median"},
                },
                "scaler": {
                    "name": "standard",
                    "params": {},
                },
            },
            "windowing": {"size": 3, "stride": 1},
        },
        "thresholding": {
            "name": "constrained_f1",
            "params": {"min_recall": 0.75},
        },
    }


@pytest.fixture
def tun_cfg():
    """Tuning-specific configuration."""
    return {
        "random_state": 42,
        "metric": "pr_auc",
        "direction": "maximize",
        "n_trials": 2,
        "n_jobs": 1,
        "prep": {
            "windowing": {"size": {"choices": [1, 3]}},
        },
        "model_space": {
            "encoder_dims": {
                "choices": [[32, 16], [64, 32]],
            },
        },
        "training_space": {},
        "thresholding": {
            "name": "constrained_f1",
            "params": {"min_recall": 0.75},
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
def tuner(cfg, tun_cfg, monkeypatch):
    import network_anomaly_detection.tuning.tuner as tuner_module

    sampled_configs = [
        {
            **deepcopy(cfg),
            "models": {"encoder_dims": [32, 16]},
        },
        {
            **deepcopy(cfg),
            "models": {"encoder_dims": [64, 32]},
        },
    ]

    sampler = Mock()
    sampler.sample.side_effect = sampled_configs

    monkeypatch.setitem(
        tuner_module.TUNING_REGISTRY,
        "ae",
        lambda base_cfg, tuning_cfg: sampler,
    )

    experiment = Mock()
    experiment.run.return_value = {
        "metrics": {"pr_auc": 0.8}
    }

    # Capture configurations passed to ExperimentPipeline.__init__.
    experiment_configs = []

    def experiment_factory(cfg, logger):
        experiment_configs.append(deepcopy(cfg))
        return experiment

    monkeypatch.setattr(
        tuner_module,
        "Experiment",
        experiment_factory,
    )

    prep = Mock()
    prep.select_training_data.side_effect = lambda X, y: (X, y)
    prep_cls = Mock(return_value=prep)

    monkeypatch.setitem(
        tuner_module.PREP_REGISTRY,
        "ae",
        prep_cls,
    )

    evaluator = Mock()

    instance = Tuner(
        cfg=cfg,
        tun_cfg=tun_cfg,
        evaluator=evaluator,
        logger=Mock(),
    )

    return (
        instance,
        sampler,
        experiment,
        prep,
        prep_cls,
        experiment_configs,
    )


def test_run_optimizes_configured_metric(tuner, data):
    instance, sampler, experiment, prep, prep_cls, _ = tuner

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

    # Each trial records its effective configuration.
    for trial in study.trials:
        assert trial.user_attrs["config"]["name"] == "ae"


def test_trial_uses_sampled_config_and_parent_metadata(tuner, data):
    (
        instance,
        sampler,
        experiment,
        prep,
        prep_cls,
        experiment_configs,
    ) = tuner

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

    # Experiment execution metadata.
    assert call.kwargs["nested"] is True
    assert call.kwargs["run_name"] == "trial_0"

    assert call.kwargs["metadata"] == {
        "mlflow.parentRunId": "parent-123",
        "run_type": "tuning_trial",
        "trial_number": "0",
        "tuning_metric": "pr_auc",
    }

    # Prep and evaluator are passed to run().
    assert call.kwargs["prep"] is prep
    assert call.kwargs["evaluator"] is instance.evaluator

    # Sampled configuration is passed to ExperimentPipeline constructor.
    assert len(experiment_configs) == 1
    assert experiment_configs[0]["models"]["encoder_dims"] == [32, 16]

    # Training data.
    np.testing.assert_array_equal(
        call.kwargs["X_train"],
        X_train,
    )
    np.testing.assert_array_equal(
        call.kwargs["y_train"],
        y_train,
    )

    # Validation data remains unfiltered.
    np.testing.assert_array_equal(
        call.kwargs["X_val"],
        X_val,
    )
    np.testing.assert_array_equal(
        call.kwargs["y_val"],
        y_val,
    )


def test_tuning_does_not_mutate_base_config(tuner, data):
    instance, _, _, _, _, _ = tuner

    original_cfg = deepcopy(instance.cfg)

    X_train, y_train, X_val, y_val = data

    instance.run(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_trials=2,
        n_jobs=1,
    )

    assert instance.cfg == original_cfg