from network_anomaly_detection.training.trainer import NNTrainer

from network_anomaly_detection.training.nn.optimizers.registry import create_optimizer
from network_anomaly_detection.training.nn.callbacks import create_callbacks
from network_anomaly_detection.training.nn.losses import create_loss

from network_anomaly_detection.training.nn.schemas import TrainingConfig




class TransformerTrainer(NNTrainer):

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
            scheduler=cfg.get("scheduler"),
            callbacks=callbacks,
            num_workers=cfg.get("num_workers", 0),
        )

        super().__init__(
            model=model.model,
            cfg=trainer_cfg,
        )


        # config
        self.training_config = {
            "trainer_type": self.__class__.__name__,
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "num_workers": cfg.get("num_workers", 0),
            "optimizer": {
                "name": cfg["optimizer"]["name"],
                "params": optimizer.defaults,
            },
            "loss": {
                "name": cfg["loss"]["name"],
                "params": cfg["loss"].get("params", {}),
            },
            "scheduler": cfg.get("scheduler"),
            "callbacks": cfg.get("callbacks", {}),
        }

    def get_config(self):
        return self.training_config
