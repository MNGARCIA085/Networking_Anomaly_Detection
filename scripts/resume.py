
import argparse
from pathlib import Path

import hydra
import mlflow
import torch

from network_anomaly_detection.infra.logging.mlflow_logger import MLFlowLogger
from network_anomaly_detection.data.data_module import DataModule
from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.experiment.experiment import Experiment
from network_anomaly_detection.evaluation.evaluator import Evaluator





@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default="7fb1cf1c9169479e8d660ec637077649",
        #required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default="last.ckpt",
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
    data_module = DataModule(
                train_path = "data/processed/v1/train.csv",
                val_path = "data/processed/v1/val.csv",
            )


    X_train, y_train, X_val, y_val = data_module.load_train_val()



    # --------------------------------------------------
    # Experiment
    # --------------------------------------------------

    logger = MLFlowLogger()
    
    experiment = Experiment(
        cfg=cfg,
        logger=logger,
    )


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    

    evaluator = Evaluator()

    result = experiment.resume(
        run_id,
        evaluator,
        X_train,
        y_train,
        X_val,
        y_val,
        checkpoint="last.ckpt",
        run_name=None,
    )


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