import pytest
import torch

from network_anomaly_detection.training.nn.schedulers import create_scheduler


@pytest.fixture
def optimizer():
    model = torch.nn.Linear(4, 2)
    return torch.optim.Adam(model.parameters(), lr=0.01)


def test_disabled_scheduler_returns_none(optimizer):
    scheduler = create_scheduler(None, optimizer)

    assert scheduler is None


def test_reduce_on_plateau_configuration(optimizer):
    result = create_scheduler(
        {
            "name": "reduce_on_plateau",
            "params": {
                "factor": 0.5,
                "patience": 3,
            },
        },
        optimizer,
    )

    assert result["scheduler"] is not None
    assert isinstance(
        result["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    assert result["monitor"] == "val_loss"
    assert result["interval"] == "epoch"
    assert result["frequency"] == 1


def test_step_lr_configuration(optimizer):
    result = create_scheduler(
        {
            "name": "step_lr",
            "params": {
                "step_size": 2,
                "gamma": 0.5,
            },
        },
        optimizer,
    )

    assert isinstance(
        result["scheduler"],
        torch.optim.lr_scheduler.StepLR,
    )


def test_unknown_scheduler_raises_error(optimizer):
    with pytest.raises(ValueError, match="Unknown"):
        create_scheduler(
            {
                "name": "not_a_scheduler",
                "params": {},
            },
            optimizer,
        )