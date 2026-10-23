import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping as LightningEarlyStopping,
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




class TrainingHistoryCallback(pl.Callback):
    def __init__(self):
        self.history = {}

    def _record(self, trainer, prefix):
        for name, value in trainer.callback_metrics.items():
            if name.startswith(prefix):
                self.history.setdefault(name, []).append(
                    value.detach().cpu().item()
                )

    def on_train_epoch_end(self, trainer, pl_module):
        self._record(trainer, "train_")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        self._record(trainer, "val_")



# factory


def create_callbacks(cfg, checkpoint_dir=None):
    # callbacks = []
    callbacks = [TrainingHistoryCallback()]

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



"""
ModelCheckpoint(
        dirpath="checkpoints/",
        filename="epoch-{epoch:04d}",
        every_n_epochs=10,
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_last=True,
    ),
"""