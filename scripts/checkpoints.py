


from network_anomaly_detection.infra.logging.mlflow_logger import MLFlowLogger 


class Experiment:

    def __init__(
        self,
        cfg,
        logger,
    ):
        self.cfg = cfg
        self.logger = logger
        self.run_id = None

    def start(self, run_name=None):

        run = self.logger.start_run(
            run_name=run_name,
        )

        self.run_id = run.info.run_id

        self.logger.log_tags({
            "status": "running",
        })

        return run

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



#-------------------------------------------
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from network_anomaly_detection.data.data_module import DataModule







@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    #...


    logger = MLFlowLogger()

    experiment = Experiment(
        cfg=cfg,
        logger=logger,
    )

    experiment.start(
        run_name=cfg.get("run_name"),
    )

    try:

        # --------------------------------------------------
        # DATA
        # --------------------------------------------------

        data = DataModule(
            "data/arriba.csv",
            "data/arriba.csv",
        )

        X_train, y_train, X_val, y_val = data.load()


        # --------------------------------------------------
        # PREPROCESSING
        # --------------------------------------------------

        from network_anomaly_detection.preprocessing.registry import (
            PREP_REGISTRY,
        )

        prep_cls = PREP_REGISTRY[
            cfg.model_type.name
        ]

        prep = prep_cls(
            cfg.model_type,
        )

        X_train, y_train, X_val, y_val = prep.build_prep(
            X_train,
            y_train,
            X_val,
            y_val,
        )


        # --------------------------------------------------
        # SAVE PREPROCESSING
        # --------------------------------------------------

        prep_path = logger.artifact_path(
            "prep.joblib",
            artifact_dir=(
                logger.run_artifact_dir()
                / "preprocessing"
            ),
        )

        prep.save(
            prep_path,
        )

        logger.log_artifact(
            prep_path,
            artifact_path="preprocessing",
        )


        # --------------------------------------------------
        # MODEL
        # --------------------------------------------------

        from network_anomaly_detection.models.registry import (
            MODEL_REGISTRY,
        )

        model_cls = MODEL_REGISTRY[
            cfg.model_type.name
        ]

        model = model_cls(
            cfg.model_type.models,
            input_shape=X_train.shape[1:],
        )


        # --------------------------------------------------
        # MODEL INPUT ADAPTATION
        # --------------------------------------------------

        X_train = model.adapt_input(
            X_train,
        )

        X_val = model.adapt_input(
            X_val,
        )


        # --------------------------------------------------
        # TRAINING
        # --------------------------------------------------

        from network_anomaly_detection.training.registry import (
            TRAINER_REGISTRY,
        )

        trainer_cls = TRAINER_REGISTRY[
            cfg.model_type.name
        ]

        trainer = trainer_cls(
            model=model,
            cfg=cfg.model_type.training,
            checkpoint_dir=logger.checkpoint_dir(),
        )


        trainer.fit(
            X_train,
            y_train,
            X_val,
            y_val,
        )


        # --------------------------------------------------
        # HISTORY
        # --------------------------------------------------

        logger.log_training_history(
            trainer.history,
        )


        experiment.finish()


    except Exception as e:

        experiment.fail(e)

        raise



if __name__=="__main__":
    main()