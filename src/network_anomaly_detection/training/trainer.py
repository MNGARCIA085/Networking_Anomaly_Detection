import lightning.pytorch as pl
from torch.utils.data import DataLoader

from network_anomaly_detection.training.nn.dataset import AnomalyDataset
from network_anomaly_detection.training.nn.callbacks import TrainingHistoryCallback


from  network_anomaly_detection.training.nn.schedulers import create_scheduler



class AnomalyLightningModule(pl.LightningModule):

    def __init__(self, model, optimizer, criterion, scheduler_cfg=None):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler_cfg = scheduler_cfg

    def _compute_loss(self, batch):
        X, y = batch
        recon = self.model(X)
        return self.criterion(recon, X)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch[0].size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch[0].size(0),
        )

        return loss

    def configure_optimizers(self):
        scheduler_config = create_scheduler(
            self.scheduler_cfg,
            self.optimizer,
        )

        if scheduler_config is None:
            return self.optimizer

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_config["scheduler"],
                "monitor": scheduler_config["monitor"],
                "interval": scheduler_config["interval"],
                "frequency": scheduler_config["frequency"],
            },
        }



class NNTrainer:

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.callbacks = cfg.callbacks or []
        self.history = None
        self.lightning_trainer = None

        self.module = AnomalyLightningModule(
            model=model,
            optimizer=cfg.optimizer,
            criterion=cfg.loss,
            scheduler_cfg=cfg.scheduler,
        )

    def build_dataloader(self, X, y=None, shuffle=False):
        return DataLoader(
            AnomalyDataset(X, y),
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
        )

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        ckpt_path=None,
    ):
        train_loader = self.build_dataloader(
            X_train, y_train, shuffle=False
        )

        val_loader = (
            self.build_dataloader(X_val, y_val, shuffle=False)
            if X_val is not None
            else None
        )

        trainer = pl.Trainer(
            max_epochs=self.cfg.epochs,
            callbacks=self.callbacks,
            accelerator="auto",
            devices=1,
            enable_checkpointing=True,
            logger=False, # i will log with MLFlow
        )

        trainer.fit(
            self.module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )

        self.lightning_trainer = trainer
        self.history = next(
            (
                cb.history
                for cb in self.callbacks
                if isinstance(cb, TrainingHistoryCallback)
            ),
            None,
        )

        return self.model


