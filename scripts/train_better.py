# experiment.py

from network_anomaly_detection.models.registry import MODEL_REGISTRY
from network_anomaly_detection.training.registry import TRAINER_REGISTRY


class ExperimentPipeline:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def run(
        self,
        prep,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        # Prep is already fitted by the caller.
        X_train, y_train = prep.transform_with_labels(
            X_train,
            y_train,
        )

        X_val, y_val = prep.transform_with_labels(
            X_val,
            y_val,
        )

        self._save_prep(prep)

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model_cls = MODEL_REGISTRY[
            self.cfg.model_type.name
        ]

        model = model_cls(
            self.cfg.model_type.models,
            input_shape=X_train.shape[1:],
        )

        # Model-specific input adaptation.
        X_train = model.adapt_input(X_train)
        X_val = model.adapt_input(X_val)

        # --------------------------------------------------
        # Trainer
        # --------------------------------------------------

        trainer_cls = TRAINER_REGISTRY[
            self.cfg.model_type.name
        ]

        trainer = trainer_cls(
            model=model,
            cfg=self.cfg.model_type.training,
            checkpoint_dir=self.logger.checkpoint_dir(),
        )

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        trainer.fit(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        # --------------------------------------------------
        # Artifacts
        # --------------------------------------------------

        if trainer.history is not None:
            self.logger.log_training_history(
                trainer.history,
            )

        return {
            "prep": prep,
            "model": model,
            "trainer": trainer,
        }

    def _save_prep(self, prep):

        prep_path = self.logger.artifact_path(
            "prep.joblib",
            artifact_dir=(
                self.logger.run_artifact_dir()
                / "preprocessing"
            ),
        )

        prep.save(prep_path)

        self.logger.log_artifact(
            prep_path,
            artifact_path="preprocessing",
        )


class Experiment:

    def __init__(
        self,
        cfg,
        logger,
    ):
        self.cfg = cfg
        self.logger = logger
        self.run_id = None

        self.pipeline = ExperimentPipeline(
            cfg=cfg,
            logger=logger,
        )

    def start(self, run_name=None):

        run = self.logger.start_run(
            run_name=run_name,
        )

        self.run_id = run.info.run_id

        self.logger.log_tags({
            "status": "running",
        })

        return run

    def run(
        self,
        prep,
        X_train,
        y_train,
        X_val,
        y_val,
        run_name=None,
    ):

        self.start(
            run_name=run_name,
        )

        try:

            result = self.pipeline.run(
                prep=prep,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            self.finish()

            return result

        except Exception as error:

            self.fail(error)

            raise

    def finish(self):

        self.logger.log_tags({
            "status": "finished",
        })

        self.logger.end_run()

    def fail(self, error):

        self.logger.log_tags({
            "status": "failed",
            "error": str(error),
        })

        self.logger.end_run()




import hydra
from omegaconf import DictConfig

from network_anomaly_detection.data.data_module import DataModule
from network_anomaly_detection.preprocessing.registry import PREP_REGISTRY
from network_anomaly_detection.infra.logging.mlflow_logger import MLFlowLogger
#from network_anomaly_detection.experiment import Experiment


@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    data = DataModule(
        "data/arriba.csv",
        "data/arriba.csv",
    )

    X_train, y_train, X_val, y_val = data.load()

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    prep_cls = PREP_REGISTRY[
        cfg.model_type.name
    ]

    prep = prep_cls(
        cfg.model_type,
    )

    # Fit preprocessing ONLY on training data.
    prep.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------
    # Experiment
    # --------------------------------------------------

    logger = MLFlowLogger()

    experiment = Experiment(
        cfg=cfg,
        logger=logger,
    )

    result = experiment.run(
        prep=prep,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        run_name=f"{cfg.model_type.name}_baseline",
    )

    return result


if __name__ == "__main__":
    main()