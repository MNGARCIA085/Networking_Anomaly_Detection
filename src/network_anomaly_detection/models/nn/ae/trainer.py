from network_anomaly_detection.training.nn.optimizers.registry import create_optimizer
from network_anomaly_detection.training.nn.losses import create_loss
from network_anomaly_detection.training.nn.callbacks.registry import create_callbacks
from network_anomaly_detection.training.nn.schemas import TrainingConfig

from network_anomaly_detection.training.nn.trainer import NNTrainer


"""
for ckpoint dir
from pathlib import Path
import mlflow


run_dir = Path(
    mlflow.get_artifact_uri()
).parent

checkpoint_dir = run_dir / "checkpoints"

callbacks = create_callbacks(
    cfg_training["callbacks"],
    checkpoint_dir=checkpoint_dir,
)
"""




class AETrainer(NNTrainer):

    def __init__(
        self,
        model, # model -> AEModel, model.model -> pytorch Model 
        cfg,
        #checkpoint_dir=None,
    ):
        optimizer = create_optimizer(
            cfg["optimizer"],
            model.model.parameters(),
        )

        loss = create_loss(
            cfg["loss"],
        )

        callbacks = create_callbacks(
            cfg.get("callbacks", {}),
        )
        # 
        #  checkpoint_dir=checkpoint_dir,
        # maybe add checkpoint dir!!!




        trainer_cfg = TrainingConfig(
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            optimizer=optimizer,
            loss=loss,
            callbacks=callbacks,
        )

        super().__init__(
            model=model.model,
            cfg=trainer_cfg,
        )


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