
import argparse
from pathlib import Path

import hydra
import mlflow
import torch

from network_anomaly_detection.data.data_module import DataModule
from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.models.registry import MODEL_REGISTRY
from network_anomaly_detection.training.registry import TRAINER_REGISTRY
from network_anomaly_detection.infra.logging.mlflow_logger import MLFlowLogger


def get_checkpoint(run_id, checkpoint_name="best.pt"):

    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)

    artifact_uri = run.info.artifact_uri

    checkpoint_path = (
        Path(artifact_uri.replace("file://", ""))
        / "checkpoints"
        / checkpoint_name
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    return checkpoint_path


@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default="d18e0d3181074088a1a69c426eb2a697",
        #required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default="best.pt",
    )

    args = parser.parse_args()

    run_id = args.run_id



    # --------------------------------------------------
    # Load previous MLflow run
    # --------------------------------------------------

    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)

    print(
        f"Resuming run: {run_id}"
    )

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    data = DataModule(
        "data/arriba.csv",
        "data/arriba.csv",
    )

    X_train, y_train, X_val, y_val = data.load()

    # --------------------------------------------------
    # Load fitted preprocessing
    # --------------------------------------------------

    prep_path = (
        Path(
            run.info.artifact_uri.replace(
                "file://",
                "",
            )
        )
        / "preprocessing"
        / "prep.joblib"
    )

    if not prep_path.exists():
        raise FileNotFoundError(
            f"Preprocessing artifact not found: {prep_path}"
        )

    prep = BasePrep.load(
        prep_path
    )



    # --------------------------------------------------
    # Transform data
    # --------------------------------------------------

    X_train, y_train = prep.transform_with_labels(
        X_train,
        y_train,
    )

    X_val, y_val = prep.transform_with_labels(
        X_val,
        y_val,
    )



    print('so far ok')


    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model_cls = MODEL_REGISTRY[
        cfg.model_type.name
    ]

    model = model_cls(
        cfg.model_type.models,
        input_shape=X_train.shape[1:],
    )




    X_train = model.adapt_input(
        X_train
    )

    X_val = model.adapt_input(
        X_val
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------



    logger = MLFlowLogger()


    trainer_cls = TRAINER_REGISTRY[
        cfg.model_type.name
    ]


    print('why', trainer_cls)


    #print(logger.checkpoint_dir())
    checkpoint_dir = (
        logger.run_artifact_dir_by_id(run_id)
        / "checkpoints"
    )

    trainer = trainer_cls(
        model=model,
        cfg=cfg.model_type.training,
        checkpoint_dir=checkpoint_dir,
    )





    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    checkpoint_path = get_checkpoint(
        run_id,
        args.checkpoint,
    )

    checkpoint = torch.load(
        checkpoint_path,
        #map_location=cfg.model_type.training.device,
    )

    model.model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    trainer.optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = (
        checkpoint["epoch"] + 1
    )

    print(
        f"Loaded checkpoint: {checkpoint_path}"
    )

    print(
        f"Resuming from epoch {start_epoch}"
    )


    # later will be preferable to use experiment


    # --------------------------------------------------
    # Continue training
    # --------------------------------------------------

    trainer.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        start_epoch=start_epoch,
    )

    print("Resume training finished.")


if __name__ == "__main__":
    main()






# python resume.py --run-id <RUN_ID> --checkpoint epoch_0020.pt




"""
run_id
  ↓
load original prep
  ↓
load checkpoint
  ↓
recreate model with same processed input shape
  ↓
recreate trainer/optimizer
  ↓
restore model + optimizer + epoch
  ↓
continue training

"""


"""
new AEModel
    ↓
new PyTorch AE
    ↓
new optimizer
    ↓
load checkpoint
    ↓
overwrite model weights
    ↓
overwrite optimizer state
"""