
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





@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default="a01636f58f0f47f8a4983d21f40719b0",
        #default="cb96d7f12a3648638932c3a0f76d71ce",
        #default="291831267ea24f1da923c0cd0466f138",
        #default="d18e0d3181074088a1a69c426eb2a697",
        #required=True,
    )

    parser.add_argument(
        "--checkpoint",
        #default="best.pt",
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


    """
    data = DataModule(
        "data/arriba.csv",
        "data/arriba.csv",
    )
    """

    data = DataModule(
        "data/X_train.csv",
        "data/X_test.csv",
    )

    X_train, y_train, X_val, y_val = data.load()



    print(y_val)



    from network_anomaly_detection.experiment.experiment import Experiment
    logger = MLFlowLogger()
    
    experiment = Experiment(
        cfg=cfg,
        logger=logger,
    )


    from network_anomaly_detection.evaluation.evaluator import Evaluator

    evaluator = Evaluator()

    result = experiment.resume(
        run_id,
        evaluator,
        X_train,
        y_train,
        X_val,
        y_val,
        #checkpoint="best.pt",
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