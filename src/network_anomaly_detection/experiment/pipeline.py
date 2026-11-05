from copy import deepcopy
from pathlib import Path

import mlflow
import yaml

from network_anomaly_detection.models.registry import MODEL_REGISTRY
from network_anomaly_detection.training.registry import TRAINER_REGISTRY
from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.thresholding.thresholding import Thresholding


def build_model(name, cfg, input_shape):
    """Construct a model using the configured model registry key."""
    model_cls = MODEL_REGISTRY[name]
    return model_cls(
        cfg,
        input_shape=input_shape,
    )


def build_trainer(name, model, cfg, checkpoint_dir):
    """Construct the trainer registered for the configured model type."""
    trainer_cls = TRAINER_REGISTRY[name]
    return trainer_cls(
        model=model,
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
    )


class ExperimentPipeline:
    """Orchestrates an experiment; Lightning details stay inside the trainer."""

    def __init__(self, cfg, logger):
        self.cfg = deepcopy(cfg)
        self.logger = logger
        self.run_id = None


    def _fit_thresholding(self, cfg, scores, y_val): # model_cfg
        """Fit and persist a thresholding strategy, if configured."""

        if not cfg:
            return None, None

        thresholding = Thresholding(cfg)
        thresholding.fit(scores=scores, y_val=y_val)
        threshold = thresholding.get_threshold()

        path = self.logger.artifact_path("thresholding/thresholding.pkl")
        thresholding.save(path)

        return thresholding, threshold

    def _evaluate(self, model, evaluator, thresholding_cfg, X_val, y_val):
        """Score validation data, select a threshold, and compute metrics."""
        scores = model.score(X_val)

        thresholding, threshold = self._fit_thresholding(
            cfg=thresholding_cfg,
            scores=scores,
            y_val=y_val,
        )

        predictions = model.predict(X_val, threshold=threshold)
        metrics = evaluator.evaluate(
            scores=scores,
            y_true=y_val,
            predictions=predictions,
        )

        return {
            "scores": scores,
            "predictions": predictions,
            "thresholding": thresholding,
            "threshold": threshold,
            "metrics": metrics,
        }

    def run(
        self,
        prep,
        evaluator,
        X_train,
        y_train,
        X_val,
        y_val,
        nested=False,
        run_name=None,
        metadata=None,
    ):


        cfg = self.cfg
        metadata = metadata or {}

        # tags
        self.logger.log_tags({
            "run_type": "train",
            "model_type": cfg["name"],
            "status": "running",
        })

        # common and important params
        self.logger.log_params({
            "model": cfg["name"],
        })

        # metadata
        self.logger.log_tags(metadata)

        # Preprocessing: fit should already have happened when prep was created.
        X_train, y_train = prep.transform_with_labels(X_train, y_train)
        X_val, y_val = prep.transform_with_labels(X_val, y_val)
        self.logger.save_prep(prep)


        mlflow.log_dict(
            prep.get_config(),
            artifact_file="prep_config.json",
        )


        # Keep the full validation set; derive a separate subset for training callbacks.
        # for ex, for aes for early stopping i want val with only normal data
        # val_loss should be the metric for the callbacks
        X_val_train, y_val_train = prep.select_training_data(X_val, y_val)



        # to verify: X_train -> (batch size, window_size (seq_len), num_features)


        # Model
        model = build_model(
            name=cfg['name'],
            cfg=cfg['models'],
            input_shape=X_train.shape[1:],
        )
        X_train = model.adapt_input(X_train)
        X_val = model.adapt_input(X_val)
        X_val_train = model.adapt_input(X_val_train) # early stopping for aes
                    # with only normal data


        mlflow.log_dict(
            model.get_config(),
            artifact_file="model_config.json",
        )


        # Trainer; the trainer owns the framework-specific training details.
        trainer = build_trainer(
            name=cfg['name'],
            model=model,
            cfg=cfg['training'],
            checkpoint_dir=self.logger.checkpoint_dir(),
        )
        trainer.fit(X_train, y_train, X_val_train, y_val_train)



        mlflow.log_dict(
            trainer.get_config(),
            artifact_file="trainer_config.json",
        )


        # Evaluation
        evaluation = self._evaluate(
            model=model,
            evaluator=evaluator,
            thresholding_cfg=cfg.get("thresholding", None),
            X_val=X_val,
            y_val=y_val,
        )
        val_scores = evaluation["scores"]
        predictions = evaluation["predictions"]
        thresholding = evaluation["thresholding"]
        threshold = evaluation["threshold"]
        metrics = evaluation["metrics"]



        if thresholding:
            mlflow.log_dict(
                thresholding.get_config(),
                "threshold_config.json",
            )

            self.logger.log_params({
                "threshold": evaluation["threshold"],
            })


        self.logger.log_metrics(metrics)

        if trainer.history:
            self.logger.log_training_history(trainer.history)

        # diagnosis
        self.logger.log_evaluation_diagnostics(
            y_true=y_val,
            scores=val_scores,
            predictions=predictions,
        )

        self.logger.log_validation_scores(
            y_true=y_val,
            scores=val_scores,
            predictions=predictions,
        )

        self.logger.log_candidate_model(model, metrics, cfg["name"])
        self.logger.log_tags({"status": "finished"})

        return {
            "prep": prep,
            "model": model,
            "trainer": trainer,
            "thresholding": thresholding,
            "threshold": threshold,
            "metrics": metrics,
            "config": deepcopy(cfg),
        }

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
        """Resume training from a Lightning checkpoint in an existing MLflow run."""

        cfg = self.cfg
        run = None
        run_status = "FAILED"

        try:
            # ---------------------------------------------------------
            # 1. Restore preprocessing and locate the parent checkpoint
            # ---------------------------------------------------------
            prep_path = self.logger.get_run_artifact_path(
                run_id,
                "preprocessing/prep.pkl",  # joblib
            )
            prep = BasePrep.load(prep_path)

            checkpoint_path = Path(
                self.logger.get_checkpoint_path(run_id, checkpoint)
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}"
                )

            if checkpoint_path.suffix != ".ckpt":
                raise ValueError(
                    f"Expected a Lightning .ckpt file, got: {checkpoint_path}"
                )

            # ---------------------------------------------------------
            # 2. Start a new MLflow run for the resumed experiment
            # ---------------------------------------------------------
            run = self.logger.start_run(
                run_name=run_name or f"resume_{run_id}"
            )
            self.run_id = run.info.run_id

            self.logger.log_tags({
                "status": "running",
                "run_type": "resume",
                "parent_run_id": run_id,
                "resumed_from_checkpoint": checkpoint_path.name,
            })

            # self._log_config(cfg)

            # ---------------------------------------------------------
            # 3. Apply the fitted preprocessing
            # ---------------------------------------------------------

            # only normal data for AEs
            X_train, y_train = prep.select_training_data(X_train, y_train)

            X_train, y_train = prep.transform_with_labels(X_train, y_train)
            X_val, y_val = prep.transform_with_labels(X_val, y_val)


            X_val_train, y_val_train = prep.select_training_data(X_val, y_val)



            # ---------------------------------------------------------
            # 4. Reconstruct the model and adapt its input
            # ---------------------------------------------------------
            model = build_model(
                name=cfg['name'],
                cfg=cfg['models'],
                input_shape=X_train.shape[1:],
            )

            X_train = model.adapt_input(X_train)
            X_val = model.adapt_input(X_val)

            X_val_train = model.adapt_input(X_val_train) # new!!

            # ---------------------------------------------------------
            # 5. Resume Lightning training
            # ---------------------------------------------------------
            trainer = build_trainer(
                name=cfg['name'],
                model=model,
                cfg=cfg['training'],
                checkpoint_dir=self.logger.checkpoint_dir(),
            )

            trainer.fit(
                X_train,
                y_train,
                X_val_train,
                y_val_train,
                ckpt_path=str(checkpoint_path),
            )

            # IMPORTANT:
            # NNTrainer.fit() must transfer the restored LightningModule
            # weights back to `model` before returning, if the Lightning
            # module and `model` are separate objects.

            # ---------------------------------------------------------
            # 6. Score validation data and fit thresholding
            # ---------------------------------------------------------
            evaluation = self._evaluate(
                model=model,
                evaluator=evaluator,
                thresholding_cfg=cfg.get("thresholding", None),
                X_val=X_val,
                y_val=y_val,
            )
            val_scores = evaluation["scores"]
            predictions = evaluation["predictions"]
            thresholding = evaluation["thresholding"]
            threshold = evaluation["threshold"]
            metrics = evaluation["metrics"]

            # ---------------------------------------------------------
            # 7. Evaluate and log metrics
            # ---------------------------------------------------------
            self.logger.log_metrics(metrics)

            # ---------------------------------------------------------
            # 8. Log training history and diagnostics
            # ---------------------------------------------------------
            if trainer.history:
                self.logger.log_training_history(trainer.history)

            
            # diagnosis
            self.logger.log_evaluation_diagnostics(
                y_true=y_val,
                scores=val_scores,
                predictions=predictions,
            )

            self.logger.log_validation_scores(
                y_true=y_val,
                scores=val_scores,
                predictions=predictions,
            )



            self.logger.log_candidate_model(model, metrics, cfg["name"])
            self.logger.log_tags({"status": "finished"})
            run_status = "FINISHED"

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


