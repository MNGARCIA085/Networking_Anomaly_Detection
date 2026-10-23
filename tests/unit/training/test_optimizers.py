from types import SimpleNamespace

import optuna
import pytest
import torch

from network_anomaly_detection.training.nn.optimizers.registry import (
    create_optimizer,
)
from network_anomaly_detection.training.nn.optimizers.implementations import (
    AdamOptimizer,
    SGDOptimizer,
)


@pytest.fixture
def parameters():
    model = torch.nn.Linear(4, 2)
    return model.parameters()


def test_create_adam_optimizer(parameters):
    cfg = {
        "name": "adam",
        "params": {
            "lr": 0.001,
            "betas": (0.9, 0.999),
        },
    }

    optimizer = create_optimizer(cfg, parameters)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == pytest.approx(0.001)
    assert optimizer.defaults["betas"] == (0.9, 0.999)


def test_create_sgd_optimizer(parameters):
    cfg = {
        "name": "sgd",
        "params": {
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        },
    }

    optimizer = create_optimizer(cfg, parameters)

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["lr"] == pytest.approx(0.01)
    assert optimizer.defaults["momentum"] == pytest.approx(0.9)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.0001)


def test_adam_optimizer_sampling():
    study = optuna.create_study()
    trial = study.ask()

    cfg = SimpleNamespace(
        lr=SimpleNamespace(low=1e-4, high=1e-2, log=True),
        betas=SimpleNamespace(choices=[(0.9, 0.999), (0.8, 0.99)]),
    )

    sampled = AdamOptimizer.sample(trial, cfg)

    assert sampled["name"] == "adam"
    assert 1e-4 <= sampled["params"]["lr"] <= 1e-2
    assert sampled["params"]["betas"] in cfg.betas.choices


def test_sgd_optimizer_sampling():
    study = optuna.create_study()
    trial = study.ask()

    cfg = SimpleNamespace(
        lr=SimpleNamespace(low=1e-4, high=1e-2, log=True),
        momentum=SimpleNamespace(low=0.0, high=0.99),
        weight_decay=SimpleNamespace(low=1e-6, high=1e-3, log=True),
    )

    sampled = SGDOptimizer.sample(trial, cfg)

    assert sampled["name"] == "sgd"
    assert 1e-4 <= sampled["params"]["lr"] <= 1e-2
    assert 0.0 <= sampled["params"]["momentum"] <= 0.99
    assert 1e-6 <= sampled["params"]["weight_decay"] <= 1e-3