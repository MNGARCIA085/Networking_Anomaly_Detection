
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)


class PrintLossCallback(pl.Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")

        msg = f"Epoch {trainer.current_epoch + 1}"

        if train_loss is not None:
            msg += f" - Train Loss: {train_loss.item():.4f}"

        if val_loss is not None:
            msg += f" - Val Loss: {val_loss.item():.4f}"

        print(msg)


from lightning.pytorch.callbacks import (
    EarlyStopping as LightningEarlyStopping,
    ModelCheckpoint,
)

def create_callbacks(cfg, checkpoint_dir=None):
    callbacks = []

    for item in cfg or []:
        name = item["name"]
        params = item.get("params", {})

        if name == "print_loss":
            callbacks.append(PrintLossCallback())

        elif name == "early_stopping":
            callbacks.append(
                LightningEarlyStopping(
                    monitor=params.get("monitor", "val_loss"),
                    mode=params.get("mode", "min"),
                    patience=params.get("patience", 5),
                )
            )

    

        elif name == "checkpoint":
            callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    monitor=params.get("monitor", "val_loss"),
                    mode=params.get("mode", "min"),
                    save_top_k=1 if params.get("save_best", True) else 0,
                    save_last=True,
                    every_n_epochs=params.get("every_n_epochs", 1),
                )
            )

    return callbacks


from network_anomaly_detection.training.trainer import NNTrainer


from network_anomaly_detection.training.nn.optimizers.registry import create_optimizer
from network_anomaly_detection.training.nn.losses import create_loss
#from network_anomaly_detection.training.nn.callbacks.registry import create_callbacks
from network_anomaly_detection.training.nn.schemas import TrainingConfig

#from network_anomaly_detection.training.nn.trainer import NNTrainer


class AETrainer(NNTrainer):

    def __init__(
        self,
        model,  # AEModel wrapper; model.model is the torch.nn.Module
        cfg,
        checkpoint_dir=None,
    ):
        optimizer = create_optimizer(
            cfg["optimizer"],
            model.model.parameters(),
        )

        loss = create_loss(cfg["loss"])

        callbacks = create_callbacks(
            cfg.get("callbacks", {}),
            checkpoint_dir=checkpoint_dir,
        )

        trainer_cfg = TrainingConfig(
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            optimizer=optimizer,
            loss=loss,
            callbacks=callbacks,
            num_workers=cfg.get("num_workers", 0),
        )

        super().__init__(
            model=model.model,
            cfg=trainer_cfg,
        )



"""
class VAELightningModule(AnomalyLightningModule):

    def _compute_loss(self, batch):
        X, y = batch

        recon, mu, logvar = self.model(X)

        reconstruction_loss = self.criterion(recon, X)

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return reconstruction_loss + kl_loss
"""