from copy import deepcopy

import mlflow
import torch
import yaml

from network_anomaly_detection.models.registry import MODEL_REGISTRY
from network_anomaly_detection.training.registry import TRAINER_REGISTRY
from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.thresholding.thresholding import Thresholding


class ExperimentPipeline:

    def __init__(self, cfg, logger):
        self.cfg = deepcopy(cfg)
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

        cfg = self.cfg
        model_cfg = cfg["model_type"]

        # --------------------------------------------------
        # Configuration and logging
        # --------------------------------------------------


        # tags
        run_type = 'train'
        self.logger.log_tags({
            "run_type": run_type,
            "model_type": model_cfg["name"],
        })


        self.logger.log_params({
            "model": model_cfg["name"],
            "random_state": cfg["random_state"],
            "window_size": model_cfg["data"]["windowing"]["size"],
            "window_stride": model_cfg["data"]["windowing"]["stride"],
        })

        #self.logger.log_text(
        #    yaml.safe_dump(cfg, sort_keys=False),
        #    "config.yaml",
        #)

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        X_train, y_train = prep.transform_with_labels(
            X_train, y_train
        )

        X_val, y_val = prep.transform_with_labels(
            X_val, y_val
        )

        self._save_prep(prep)

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model_cls = MODEL_REGISTRY[model_cfg["name"]]

        model = model_cls(
            model_cfg["models"],
            input_shape=X_train.shape[1:],
        )

        X_train = model.adapt_input(X_train)
        X_val = model.adapt_input(X_val)

        # --------------------------------------------------
        # Trainer
        # --------------------------------------------------

        trainer_cls = TRAINER_REGISTRY[model_cfg["name"]]

        trainer = trainer_cls(
            model=model,
            cfg=model_cfg["training"],
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


        thresholding_cfg = cfg["model_type"].get("thresholding")

        if thresholding_cfg:
            thresholding = Thresholding(thresholding_cfg)

            thresholding.fit(
                scores=val_scores,
                y_val=y_val,
            )

            threshold = thresholding.get_threshold()


            # save fitted thresholder
            path = self.logger.artifact_path(
                "thresholding/thresholding.pkl"
            )
            thresholding.save(path)

        else:
            thresholding = None
            threshold = None




        predictions = model.predict(
            X_val,
            threshold=threshold,
        )

        metrics = evaluator.evaluate(
            scores=val_scores,
            y_true=y_val,
            predictions=predictions,
        )

        print(metrics)

        self.logger.log_metrics(metrics)

        # --------------------------------------------------
        # Artifacts
        # --------------------------------------------------

        if trainer.history is not None:
            self.logger.log_training_history(trainer.history)

        

        #.... log model if its good enough
        self.logger.log_candidate_model(
                model,
                metrics,
                model_cfg["name"]
            )



        return {
            "prep": prep,
            "model": model,
            "trainer": trainer,
            "thresholding": thresholding,
            "threshold": threshold,
            "metrics": metrics,
            "config": deepcopy(cfg),
        }

    # ------------------------------------------------------
    # Resume training
    # ------------------------------------------------------

    def resume(
        self,
        run_id,
        evaluator,
        X_train,
        y_train,
        X_val,
        y_val,
        checkpoint=None,
        run_name=None,
    ):

        cfg = self.cfg
        model_cfg = cfg["model_type"]

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
        # Create linked MLflow run
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
            # ----------------------------------------------
            # Preprocessing
            # ----------------------------------------------

            X_train, y_train = prep.transform_with_labels(
                X_train, y_train
            )

            X_val, y_val = prep.transform_with_labels(
                X_val, y_val
            )

            # ----------------------------------------------
            # Model
            # ----------------------------------------------

            model_cls = MODEL_REGISTRY[model_cfg["name"]]

            model = model_cls(
                model_cfg["models"],
                input_shape=X_train.shape[1:],
            )

            X_train = model.adapt_input(X_train)
            X_val = model.adapt_input(X_val)

            # ----------------------------------------------
            # Trainer
            # ----------------------------------------------

            trainer_cls = TRAINER_REGISTRY[model_cfg["name"]]

            trainer = trainer_cls(
                model=model,
                cfg=model_cfg["training"],
                checkpoint_dir=self.logger.checkpoint_dir(),
            )

            # ----------------------------------------------
            # Restore state
            # ----------------------------------------------

            model.model.load_state_dict(
                state["model_state_dict"]
            )

            trainer.optimizer.load_state_dict(
                state["optimizer_state_dict"]
            )

            # ----------------------------------------------
            # Continue training
            # ----------------------------------------------

            trainer.fit(
                X_train,
                y_train,
                X_val,
                y_val,
                start_epoch=start_epoch,
            )

            # ----------------------------------------------
            # Evaluation
            # ----------------------------------------------


            

            scores = model.score(X_val)

            
            thresholding_cfg = cfg["model_type"].get("thresholding")

            

            if thresholding_cfg:
                thresholding = Thresholding(thresholding_cfg)
                thresholding.fit(scores=scores, y_val=y_val)
                threshold = thresholding.get_threshold()
                # save fitted thresholder
                path = self.logger.artifact_path(
                    "thresholding/thresholding.pkl"
                )
                thresholding.save(path)
            else:
                thresholding = None
                threshold = None


            predictions = model.predict(
                X_val,
                threshold=threshold,
            )

            metrics = evaluator.evaluate(
                scores=scores,
                y_true=y_val,
                predictions=predictions,
            )

            self.logger.log_metrics(metrics)

            #self.logger.log_text(
            #    yaml.safe_dump(cfg, sort_keys=False),
            #    "config.yaml",
            #)

            if trainer.history is not None:
                self.logger.log_training_history(trainer.history)



            #.... log model if its good enough
            self.logger.log_candidate_model(
                    model,
                    metrics,
                    model_cfg["name"]
                )



            return {
                "prep": prep,
                "model": model,
                "trainer": trainer,
                "thresholding": thresholding,
                "threshold": threshold,
                "metrics": metrics,
                "config": deepcopy(cfg),
            }

        except Exception as error:
            self.logger.log_tags({
                "status": "failed",
                "error": str(error),
            })
            raise

    # ------------------------------------------------------
    # Save preprocessing
    # ------------------------------------------------------

    def _save_prep(self, prep):

        prep_path = self.logger.artifact_path(
            "prep.pkl",
            artifact_dir=(
                self.logger.run_artifact_dir() / "preprocessing"
            ),
        )

        prep.save(prep_path)

        self.logger.log_artifact(
            prep_path,
            artifact_path="preprocessing",
        )