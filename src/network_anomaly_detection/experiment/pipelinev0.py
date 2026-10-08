
from network_anomaly_detection.models.registry import MODEL_REGISTRY
from network_anomaly_detection.training.registry import TRAINER_REGISTRY
from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.infra.logging.mlflow_logger import MLFlowLogger

import mlflow
import torch



from network_anomaly_detection.thresholding.thresholding import Thresholding




class ExperimentPipeline:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def run(
        self,
        prep,
        evaluator,
        X_train,
        y_train,
        X_val,
        y_val,
    ):


        #print(self.cfg)

        # TAGS!!!!

        # MOST important params
        self.logger.log_params({
            "model": self.cfg.model_type.name,
            "random_state": self.cfg.random_state,
            "window_size": self.cfg.model_type.prep.windowing.size,
            "window_stride": self.cfg.model_type.prep.windowing.stride,
            #"optimizer": self.cfg.model_type.training.optimizer.name,
            #"learning_rate": self.cfg.model_type.training.optimizer.params.lr,
            #"epochs": self.cfg.model_type.training.epochs,
            #"batch_size": self.cfg.model_type.training.batch_size,
        })


        # later save all confis as artiufac
        
        """
        from omegaconf import OmegaConf
        self.logger.log_text(
            OmegaConf.to_yaml(
                self.cfg,
                resolve=True,
            ),
            "config.yaml",
        )
        """




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
        # Evaluation
        # --------------------------------------------------

        val_scores = model.score(X_val)

        thresholding_cfg = self.cfg.model_type.get("thresholding")

        if thresholding_cfg:
            thresholding = Thresholding(thresholding_cfg)

            thresholding.fit(
                scores=val_scores,
                y_val=y_val, # y_true=y_val later
            )

            threshold = thresholding.get_threshold()
        else:
            thresholding = None
            threshold = None

        


        # later -> save theshold as an artifact


        predictions = model.predict(
            X_val,
            threshold=threshold,
        )



        metrics = evaluator.evaluate(
            scores=val_scores,
            y_true=y_val,
            predictions=predictions,
        )
       

        self.logger.log_metrics(metrics)


        # --------------------------------------------------
        # Artifacts
        # --------------------------------------------------

        if trainer.history is not None:
            self.logger.log_training_history(
                trainer.history,
            )


        print(metrics)

        return {
            "prep": prep,
            "model": model,
            "trainer": trainer,
            "metrics": metrics
        }



    #-------------------------resume training-------------------#
    def resume(
        self,
        run_id,
        X_train,
        y_train,
        X_val,
        y_val,
        checkpoint=None,
        run_name=None,
    ):
        # --------------------------------------------------
        # Load preprocessing
        # --------------------------------------------------

        prep_path = self.logger.get_run_artifact_path(
            run_id,
            "preprocessing/prep.joblib",
        )

        prep = BasePrep.load(prep_path)

        # --------------------------------------------------
        # Load checkpoint
        # --------------------------------------------------

        checkpoint_path = self.logger.get_checkpoint_path(
            run_id,
            checkpoint,
        )

        state = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        start_epoch = state["epoch"] + 1

        # --------------------------------------------------
        # New MLflow run
        # --------------------------------------------------

        run = self.logger.start_run(
            run_name=run_name or f"resume_{run_id}",
        )

        self.run_id = run.info.run_id

        self.logger.log_tags({
            "status": "running",
            "run_type": "resume",
            "parent_run_id": run_id,
            "resumed_from_checkpoint": checkpoint_path.name,
        })

        self.logger.log_params({
            "resume_from_epoch": start_epoch,
        })

        try:
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
            # Restore state
            # --------------------------------------------------

            model.model.load_state_dict(
                state["model_state_dict"]
            )

            trainer.optimizer.load_state_dict(
                state["optimizer_state_dict"]
            )

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

            # --------------------------------------------------
            # Training history
            # --------------------------------------------------


            # --------------------------------------------------
            # Evaluation (check)
            # --------------------------------------------------

            scores = model.score(X_val)

            predictions = model.predict(
                X_val,
                threshold=0.7, # calcuaetd later
                #threshold=self.cfg.model_type.threshold,
            )

            metrics = evaluator.evaluate(
                scores=scores,
                y_true=y_val,
                predictions=predictions,
            )


            self.logger.log_metrics(metrics)



            #.....
            if trainer.history is not None:
                self.logger.log_training_history(
                    trainer.history,
                )

            return {
                "prep": prep,
                "model": model,
                "trainer": trainer,
            }

        except Exception as error:
            self.logger.log_tags({
                "status": "failed",
                "error": str(error),
            })
            raise





    #.....................
    # dont hink it belomgs here!!!!!: move to prep probably
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








"""
Because run() means:

create a new experiment execution from the supplied prep/data.

Whereas resume() means:

recover an existing execution from a checkpoint and create a new linked MLflow run.

Those are genuinely different workflows.

class Experiment:

    def run(...):
        # normal training
        # creates new MLflow run

    def resume(...):
        # loads original run metadata
        # loads prep/checkpoint
        # creates NEW MLflow run
        # links it with parent_run_id
        # restores model + optimizer
        # continues training

"""