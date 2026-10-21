import hydra
import mlflow
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import TimeSeriesSplit

from network_anomaly_detection.data.data_module import DataModule
from network_anomaly_detection.evaluation.evaluator import Evaluator
from network_anomaly_detection.experiment.experiment import Experiment
from network_anomaly_detection.infra.logging.mlflow_logger import MLFlowLogger
from network_anomaly_detection.preprocessing.registry import PREP_REGISTRY


N_SPLITS = 2


def aggregate_fold_metrics(logger, fold_run_ids):
    fold_metrics = [
        mlflow.get_run(run_id).data.metrics
        for run_id in fold_run_ids
    ]

    if not fold_metrics:
        return

    common_metrics = set.intersection(
        *(set(metrics) for metrics in fold_metrics)
    )

    metrics_to_log = {}

    for name in sorted(common_metrics):
        values = np.asarray(
            [metrics[name] for metrics in fold_metrics],
            dtype=float,
        )

        metrics_to_log[f"cv.mean.{name}"] = float(values.mean())
        metrics_to_log[f"cv.std.{name}"] = float(values.std())

    logger.log_metrics(metrics_to_log)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    #cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Load the existing chronological development partitions.
    data = DataModule(
        train_path=to_absolute_path("data/processed/v1/train.csv"),
        val_path=to_absolute_path("data/processed/v1/val.csv"),
    )
    X_train, y_train, X_val, y_val = data.load_train_val()

    # CV uses the combined development period, in chronological order.
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    logger = MLFlowLogger(
        tracking_db=to_absolute_path(cfg.paths.mlflow_db),
        artifact_dir=to_absolute_path(cfg.paths.mlflow_artifacts),
    )

    fold_run_ids = []

    with logger.start_run(run_name="temporal_cross_validation") as parent_run:
        logger.log_tags({
            "run_type": "cross_validation",
            "status": "running",
            "cv_strategy": "expanding_window",
            "cv_n_splits": str(N_SPLITS),
        })

        try:


            cfg_dict = OmegaConf.to_container(cfg, resolve=True)


            for fold_idx, (train_idx, val_idx) in enumerate(
                splitter.split(X), start=1
            ):
                X_fold_train = X.iloc[train_idx]
                y_fold_train = y.iloc[train_idx]
                X_fold_val = X.iloc[val_idx]
                y_fold_val = y.iloc[val_idx]

                # Every fold gets a fresh preprocessing instance.
                prep_cls = PREP_REGISTRY[cfg_dict['model_type']['name']]
                prep = prep_cls(cfg_dict['model_type'])

                # Apply model-specific selection to training data only.
                X_fold_train, y_fold_train = prep.select_training_data(
                    X_fold_train, y_fold_train
                )

                # Fit preprocessing on this fold's training data.
                prep.fit(X_fold_train, y_fold_train)


                


                experiment = Experiment(
                    cfg=cfg_dict, # cfg_dict
                    logger=logger,
                )

                experiment.run(
                    prep=prep,
                    evaluator=Evaluator(),
                    X_train=X_fold_train,
                    y_train=y_fold_train,
                    X_val=X_fold_val,
                    y_val=y_fold_val,
                    run_name=f"fold_{fold_idx}",
                    nested=True,
                    metadata={
                        "cv_fold": fold_idx,
                        "train_size": len(train_idx),
                        "val_size": len(val_idx),
                    },
                )

                fold_run_ids.append(experiment.run_id)

            aggregate_fold_metrics(logger, fold_run_ids)
            logger.log_tags({"status": "finished"})

        except Exception as error:
            logger.log_tags({
                "status": "failed",
                "error": str(error),
            })
            raise


if __name__ == "__main__":
    main()