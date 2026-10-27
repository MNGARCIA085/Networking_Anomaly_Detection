import torch
import torch.nn.functional as F

from network_anomaly_detection.training.trainer import (
    AnomalyLightningModule,
)

from network_anomaly_detection.training.trainer import NNTrainer
from network_anomaly_detection.training.nn.optimizers.registry import create_optimizer
from network_anomaly_detection.training.nn.callbacks import create_callbacks
from network_anomaly_detection.training.nn.schemas import TrainingConfig



class VAELightningModule(AnomalyLightningModule):

    def __init__(
        self,
        model,
        optimizer,
        criterion=None,
        scheduler_cfg=None,
        beta=1.0,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler_cfg=scheduler_cfg,
        )

        self.beta = beta


    def _compute_loss(self, batch):
        X, _ = batch

        recon, mu, logvar = self.model(X)

        recon_loss = F.mse_loss(
            recon,
            X,
            reduction="mean",
        )

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return recon_loss + self.beta * kl_loss

    

    """
    def _compute_loss(self, batch):
        X, _ = batch

        recon, mu, logvar = self.model(X)

        recon_loss = F.mse_loss(
            recon,
            X,
            reduction="mean",
        )

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return recon_loss + self.model.config.beta * kl_loss
    """






class VAETrainer(NNTrainer):

    def __init__(
        self,
        model,
        cfg,
        checkpoint_dir=None,
    ):
        optimizer = create_optimizer(
            cfg["optimizer"],
            model.model.parameters(),
        )

        callbacks = create_callbacks(
            cfg.get("callbacks", {}),
            checkpoint_dir=checkpoint_dir,
        )

        trainer_cfg = TrainingConfig(
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            optimizer=optimizer,
            loss=None,
            scheduler=cfg.get("scheduler"),
            callbacks=callbacks,
            num_workers=cfg.get("num_workers", 0),
        )

        super().__init__(
            model=model.model,
            cfg=trainer_cfg,
            lightning_module_cls=VAELightningModule,
        )

        self.module.beta = model.model.config.beta


        self.training_config = {
            "trainer_type": self.__class__.__name__,
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "num_workers": cfg.get("num_workers", 0),
            "optimizer": {
                "name": optimizer.__class__.__name__,
                "params": optimizer.defaults.copy(),
            },
            "loss": {
                "name": "vae_loss",
                "reconstruction": {
                    "name": "MSELoss",
                    "reduction": "mean",
                },
                "kl_divergence": {
                    "name": "kl_divergence",
                    "reduction": "mean",
                },
                "beta": self.module.beta,
            },
            "scheduler": cfg.get("scheduler"),
            "callbacks": cfg.get("callbacks", {}),
        }

    def get_config(self):
        return self.training_config




# maybe later correct beta stuff