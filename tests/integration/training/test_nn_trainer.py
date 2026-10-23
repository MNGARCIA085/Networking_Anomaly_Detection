from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from network_anomaly_detection.training.trainer import NNTrainer
from network_anomaly_detection.training.nn.callbacks import (
    TrainingHistoryCallback,
)


class TinyAutoencoder(nn.Module):
    """Small reconstruction model for trainer integration tests."""

    def __init__(self, n_features=2):
        super().__init__()
        self.linear = nn.Linear(n_features, n_features)

    def forward(self, x):
        return self.linear(x)


def make_trainer(model, epochs=2, batch_size=8):
    history_callback = TrainingHistoryCallback()

    cfg = SimpleNamespace(
        epochs=epochs,
        batch_size=batch_size,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        loss=nn.MSELoss(),
        scheduler=None,
        callbacks=[history_callback],
        num_workers=0,
    )

    return NNTrainer(model=model, cfg=cfg), history_callback


def make_data(n_samples=32, seq_len=3, n_features=2):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, seq_len, n_features)).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    return X, y


def test_build_dataloader_preserves_samples_and_labels():
    model = TinyAutoencoder()
    trainer, _ = make_trainer(model)

    X, y = make_data(n_samples=16)
    loader = trainer.build_dataloader(X, y, shuffle=False)

    batches = list(loader)

    assert sum(len(batch_X) for batch_X, _ in batches) == len(X)

    batch_X, batch_y = batches[0]
    assert batch_X.shape == (8, 3, 2)
    assert batch_X.dtype == torch.float32
    assert batch_y.dtype == torch.long
    assert torch.equal(batch_y, torch.zeros(8, dtype=torch.long))


def test_build_dataloader_supports_unlabeled_data():
    model = TinyAutoencoder()
    trainer, _ = make_trainer(model)

    X, _ = make_data(n_samples=16)
    loader = trainer.build_dataloader(X, shuffle=False)

    batches = list(loader)

    assert sum(len(batch) for batch in batches) == len(X)
    assert isinstance(batches[0], torch.Tensor)
    assert batches[0].shape == (8, 3, 2)


def test_fit_completes_training_and_validation():
    model = TinyAutoencoder()
    trainer, _ = make_trainer(model, epochs=2)

    X_train, y_train = make_data(n_samples=32)
    X_val, y_val = make_data(n_samples=16)

    result = trainer.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
    )

    assert result is model
    assert trainer.lightning_trainer is not None
    assert trainer.lightning_trainer.current_epoch == 2
    assert trainer.history is not None


def test_fit_updates_model_parameters():
    model = TinyAutoencoder()
    trainer, _ = make_trainer(model, epochs=2)

    initial_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    X_train, y_train = make_data(n_samples=32)

    trainer.fit(X_train, y_train)

    assert any(
        not torch.equal(initial, updated.detach())
        for initial, updated in zip(initial_parameters, model.parameters())
    )


def test_fit_records_training_and_validation_history():
    model = TinyAutoencoder()
    trainer, history_callback = make_trainer(model, epochs=2)

    X_train, y_train = make_data(n_samples=32)
    X_val, y_val = make_data(n_samples=16)

    trainer.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
    )

    assert trainer.history is history_callback.history
    assert len(trainer.history["train_loss"]) == 2
    assert len(trainer.history["val_loss"]) == 2

    assert np.isfinite(trainer.history["train_loss"]).all()
    assert np.isfinite(trainer.history["val_loss"]).all()