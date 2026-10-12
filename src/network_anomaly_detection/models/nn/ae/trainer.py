from network_anomaly_detection.training.trainer import NNTrainer

from network_anomaly_detection.training.nn.optimizers.registry import create_optimizer
from network_anomaly_detection.training.nn.callbacks import create_callbacks
from network_anomaly_detection.training.nn.losses import create_loss

from network_anomaly_detection.training.nn.schemas import TrainingConfig




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



"""
as ref for the class
trainer_cls = TRAINER_REGISTRY[
            cfg_training["type"]
        ]

        return trainer_cls(trainer_cfg)
"""


"""
Your separation is sound:

AEModel                  # project abstraction
   │
   └── AE                 # PyTorch implementation

AETrainer
   │
   └── NNTrainer          # generic PyTorch training

The responsibilities are well separated:

AEModel: adapt_input, score, predict, model-specific inference.
AE: nn.Module, architecture/forward.
AETrainer: builds AE-specific training components.
NNTrainer: generic PyTorch training loop.
BaseModel: common anomaly-model contract, including Isolation Forest.
"""