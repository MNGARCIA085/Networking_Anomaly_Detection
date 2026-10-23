from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from network_anomaly_detection.training.nn.callbacks import (
    PrintLossCallback,
    TrainingHistoryCallback,
    create_callbacks,
)


def make_trainer(metrics=None, sanity_checking=False, current_epoch=0):
    return SimpleNamespace(
        callback_metrics=metrics or {},
        sanity_checking=sanity_checking,
        current_epoch=current_epoch,
    )


def test_history_records_training_metrics():
    callback = TrainingHistoryCallback()
    trainer = make_trainer(
        {
            "train_loss": torch.tensor(0.5),
            "train_accuracy": torch.tensor(0.8),
            "val_loss": torch.tensor(0.6),
        }
    )

    callback.on_train_epoch_end(trainer, None)

    assert callback.history == {
        "train_loss": [0.5],
        "train_accuracy": [pytest.approx(0.8)],
    }


def test_history_records_validation_metrics():
    callback = TrainingHistoryCallback()
    trainer = make_trainer(
        {
            "train_loss": torch.tensor(0.5),
            "val_loss": torch.tensor(0.4),
            "val_pr_auc": torch.tensor(0.9),
        }
    )

    callback.on_validation_epoch_end(trainer, None)

    assert callback.history["val_loss"] == [pytest.approx(0.4)]
    assert callback.history["val_pr_auc"] == [pytest.approx(0.9)]
    assert "train_loss" not in callback.history


def test_history_skips_validation_sanity_check():
    callback = TrainingHistoryCallback()
    trainer = make_trainer(
        {"val_loss": torch.tensor(0.4)},
        sanity_checking=True,
    )

    callback.on_validation_epoch_end(trainer, None)

    assert callback.history == {}


def test_history_accumulates_metrics_across_epochs():
    callback = TrainingHistoryCallback()

    callback.on_train_epoch_end(
        make_trainer({"train_loss": torch.tensor(0.8)}),
        None,
    )
    callback.on_train_epoch_end(
        make_trainer({"train_loss": torch.tensor(0.3)}, current_epoch=1),
        None,
    )

    assert callback.history["train_loss"] == [
        pytest.approx(0.8),
        pytest.approx(0.3),
    ]


def test_history_ignores_metrics_without_matching_prefix():
    callback = TrainingHistoryCallback()
    trainer = make_trainer(
        {
            "loss": torch.tensor(0.5),
            "val_loss": torch.tensor(0.4),
        }
    )

    callback.on_train_epoch_end(trainer, None)

    assert callback.history == {}


def test_print_loss_callback(capsys):
    callback = PrintLossCallback()
    trainer = make_trainer(
        {
            "train_loss": torch.tensor(0.5),
            "val_loss": torch.tensor(0.4),
        },
        current_epoch=1,
    )

    callback.on_train_epoch_end(trainer, None)

    output = capsys.readouterr().out
    assert "Epoch 2" in output
    assert "Train Loss: 0.5000" in output
    assert "Val Loss: 0.4000" in output


def test_create_callbacks_always_includes_history_callback():
    callbacks = create_callbacks(None)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], TrainingHistoryCallback)


def test_create_callbacks_builds_early_stopping():
    callbacks = create_callbacks(
        [
            {
                "name": "early_stopping",
                "params": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": 3,
                },
            }
        ]
    )

    early_stopping = next(
        callback
        for callback in callbacks
        if isinstance(callback, EarlyStopping)
    )

    assert early_stopping.monitor == "val_loss"
    assert early_stopping.mode == "min"
    assert early_stopping.patience == 3


def test_create_callbacks_uses_early_stopping_defaults():
    callbacks = create_callbacks([{"name": "early_stopping"}])

    early_stopping = next(
        callback
        for callback in callbacks
        if isinstance(callback, EarlyStopping)
    )

    assert early_stopping.monitor == "val_loss"
    assert early_stopping.mode == "min"
    assert early_stopping.patience == 5


def test_create_callbacks_builds_checkpoint(tmp_path):
    callbacks = create_callbacks(
        [
            {
                "name": "checkpoint",
                "params": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_best": True,
                    "every_n_epochs": 2,
                },
            }
        ],
        checkpoint_dir=tmp_path,
    )

    checkpoint = next(
        callback
        for callback in callbacks
        if isinstance(callback, ModelCheckpoint)
    )

    assert Path(checkpoint.dirpath) == tmp_path
    assert checkpoint.monitor == "val_loss"
    assert checkpoint.mode == "min"
    assert checkpoint.save_top_k == 1
    assert checkpoint.save_last is True
    assert checkpoint.every_n_epochs == 2


def test_checkpoint_can_disable_best_checkpoint(tmp_path):
    callbacks = create_callbacks(
        [
            {
                "name": "checkpoint",
                "params": {"save_best": False},
            }
        ],
        checkpoint_dir=tmp_path,
    )

    checkpoint = next(
        callback
        for callback in callbacks
        if isinstance(callback, ModelCheckpoint)
    )

    assert checkpoint.save_top_k == 0
    assert checkpoint.save_last is True