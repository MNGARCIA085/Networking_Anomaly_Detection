from copy import deepcopy

import yaml

from network_anomaly_detection.models.registry import MODEL_REGISTRY
from network_anomaly_detection.training.registry import TRAINER_REGISTRY
from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.thresholding.thresholding import Thresholding


class ExperimentPipeline:
    """Orchestrates an experiment; Lightning details stay inside NNTrainer."""

    def __init__(self, cfg, logger):
        self.cfg = deepcopy(cfg)
        self.logger = logger
        self.run_id = None

    def run(self, prep, evaluator, X_train, y_train, X_val, y_val):
        cfg = self.cfg
        model_cfg = cfg["model_type"]

        self.logger.log_tags({
            "run_type": "train",
            "model_type": model_cfg["name"],
            "status": "running",
        })
        self.logger.log_params({
            "model": model_cfg["name"],
            "random_state": cfg["random_state"],
            "window_size": model_cfg["data"]["windowing"]["size"],
            "window_stride": model_cfg["data"]["windowing"]["stride"],
        })
        #self._log_config(cfg)

        # Preprocessing: fit should already have happened when prep was created.
        X_train, y_train = prep.transform_with_labels(X_train, y_train)
        X_val, y_val = prep.transform_with_labels(X_val, y_val)
        self._save_prep(prep)

        # Model
        model_cls = MODEL_REGISTRY[model_cfg["name"]]
        model = model_cls(
            model_cfg["models"],
            input_shape=X_train.shape[1:],
        )
        X_train = model.adapt_input(X_train)
        X_val = model.adapt_input(X_val)

        # Trainer; NNTrainer owns the Lightning Trainer and callbacks.
        trainer_cls = TRAINER_REGISTRY[model_cfg["name"]]
        trainer = trainer_cls(
            model=model,
            cfg=model_cfg["training"],
            checkpoint_dir=self.logger.checkpoint_dir(),
        )
        trainer.fit(X_train, y_train, X_val, y_val)


        """
        import mlflow

        for filename in ("last.ckpt",):
            path = os.path.join(checkpoint_dir, filename)
            if os.path.isfile(path):
                mlflow.log_artifact(path, artifact_path="checkpoints")

        best_path = checkpoint_callback.best_model_path
        if best_path and os.path.isfile(best_path):
            mlflow.log_artifact(best_path, artifact_path="checkpoints")
        """

        # Evaluation
        val_scores = model.score(X_val)
        thresholding_cfg = model_cfg.get("thresholding")

        if thresholding_cfg:
            thresholding = Thresholding(thresholding_cfg)
            thresholding.fit(scores=val_scores, y_val=y_val)
            threshold = thresholding.get_threshold()
            #path = self.logger.artifact_path("thresholding/thresholding.pkl")
            #thresholding.save(path)
            # save fitted thresholder
            path = self.logger.artifact_path(
                "thresholding/thresholding.pkl"
            )
            thresholding.save(path)
        else:
            thresholding = None
            threshold = None

        predictions = model.predict(X_val, threshold=threshold)
        metrics = evaluator.evaluate(
            scores=val_scores,
            y_true=y_val,
            predictions=predictions,
        )
        self.logger.log_metrics(metrics)


        """
        history = trainer.history
        if history is not None:
            self.logger.log_training_history(history)

        diagnostics = collect_diagnostics(
            y_true=y_val,
            scores=val_scores,
            predictions=predictions,
            history=history,
        )
        self.logger.log_diagnostics(diagnostics)

        from dataclasses import dataclass, field

        @dataclass
        class TrainingHistory:
            metrics: dict = field(default_factory=dict)
            history = TrainingHistory(
            metrics={
                "train_loss": train_losses,
                "val_loss": val_losses,
            }
        )
        """


        self.logger.log_candidate_model(model, metrics, model_cfg["name"])
        self.logger.log_tags({"status": "finished"})

        return {
            "prep": prep,
            "model": model,
            "trainer": trainer,
            "thresholding": thresholding,
            "threshold": threshold,
            "metrics": metrics,
            #"diagnostics": diagnostics,
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
        #Resume training from a Lightning checkpoint in an existing MLflow run.
        import mlflow
        from pathlib import Path
        from copy import deepcopy

        cfg = self.cfg
        model_cfg = cfg["model_type"]
        run = None
        run_status = "FAILED"

        try:
            # ---------------------------------------------------------
            # 1. Restore preprocessing and locate the parent checkpoint
            # ---------------------------------------------------------
            prep_path = self.logger.get_run_artifact_path(
                run_id,
                "preprocessing/prep.joblib",
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

            #self._log_config(cfg)

            # ---------------------------------------------------------
            # 3. Apply the fitted preprocessing
            # ---------------------------------------------------------
            X_train, y_train = prep.transform_with_labels(
                X_train, y_train
            )
            X_val, y_val = prep.transform_with_labels(
                X_val, y_val
            )

            # ---------------------------------------------------------
            # 4. Reconstruct the model and adapt its input
            # ---------------------------------------------------------
            model_cls = MODEL_REGISTRY[model_cfg["name"]]

            model = model_cls(
                model_cfg["models"],
                input_shape=X_train.shape[1:],
            )

            X_train = model.adapt_input(X_train)
            X_val = model.adapt_input(X_val)

            # ---------------------------------------------------------
            # 5. Resume Lightning training
            # ---------------------------------------------------------
            trainer_cls = TRAINER_REGISTRY[model_cfg["name"]]

            trainer = trainer_cls(
                model=model,
                cfg=model_cfg["training"],
                checkpoint_dir=self.logger.checkpoint_dir(),
            )

            trainer.fit(
                X_train,
                y_train,
                X_val,
                y_val,
                ckpt_path=str(checkpoint_path),
            )

            # IMPORTANT:
            # NNTrainer.fit() must transfer the restored LightningModule
            # weights back to `model` before returning, if the Lightning
            # module and `model` are separate objects.

            # ---------------------------------------------------------
            # 6. Score validation data and fit thresholding
            # ---------------------------------------------------------
            scores = model.score(X_val)

            thresholding_cfg = model_cfg.get("thresholding")

            if thresholding_cfg:
                thresholding = Thresholding(thresholding_cfg)
                thresholding.fit(
                    scores=scores,
                    y_val=y_val,
                )
                threshold = thresholding.get_threshold()

                threshold_path = self.logger.artifact_path(
                    "thresholding/thresholding.pkl"
                )
                thresholding.save(threshold_path)
            else:
                thresholding = None
                threshold = None

            # ---------------------------------------------------------
            # 7. Evaluate and log metrics
            # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # 8. Log training history and diagnostics
            # ---------------------------------------------------------
            
            
            """
            history = trainer.history

            if history is not None:
                self.logger.log_training_history(history)

            diagnostics = collect_diagnostics(
                y_true=y_val,
                scores=scores,
                predictions=predictions,
                history=history,
            )

            self.logger.log_diagnostics(diagnostics)
            self.logger.log_candidate_model(
                model,
                metrics,
                model_cfg["name"],
            )
            """

            self.logger.log_tags({"status": "finished"})
            run_status = "FINISHED"

            return {
                "prep": prep,
                "model": model,
                "trainer": trainer,
                "thresholding": thresholding,
                "threshold": threshold,
                "metrics": metrics,
                #"diagnostics": diagnostics,
                "config": deepcopy(cfg),
            }

        except Exception as error:
            self.logger.log_tags({
                "status": "failed",
                "error": str(error),
            })
            raise

    def _save_prep(self, prep):
        # Keep this artifact path consistent with resume().
        prep_path = self.logger.artifact_path(
            "prep.joblib",
            artifact_dir=self.logger.run_artifact_dir() / "preprocessing",
        )
        prep.save(prep_path)
        self.logger.log_artifact(prep_path, artifact_path="preprocessing")

    """
    def _log_config(self, cfg):
        self.logger.log_text(
            yaml.safe_dump(cfg, sort_keys=False),
            "config.yaml",
        )
    """



# prep as pkl and not jobliob!!!!!!!!

"""
nw resume
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
    Resume training from a Lightning checkpoint in an existing MLflow run.
    import mlflow
    from pathlib import Path
    from copy import deepcopy

    cfg = self.cfg
    model_cfg = cfg["model_type"]
    run = None
    run_status = "FAILED"

    try:
        # ---------------------------------------------------------
        # 1. Restore preprocessing and locate the parent checkpoint
        # ---------------------------------------------------------
        prep_path = self.logger.get_run_artifact_path(
            run_id,
            "preprocessing/prep.joblib",
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

        self._log_config(cfg)

        # ---------------------------------------------------------
        # 3. Apply the fitted preprocessing
        # ---------------------------------------------------------
        X_train, y_train = prep.transform_with_labels(
            X_train, y_train
        )
        X_val, y_val = prep.transform_with_labels(
            X_val, y_val
        )

        # ---------------------------------------------------------
        # 4. Reconstruct the model and adapt its input
        # ---------------------------------------------------------
        model_cls = MODEL_REGISTRY[model_cfg["name"]]

        model = model_cls(
            model_cfg["models"],
            input_shape=X_train.shape[1:],
        )

        X_train = model.adapt_input(X_train)
        X_val = model.adapt_input(X_val)

        # ---------------------------------------------------------
        # 5. Resume Lightning training
        # ---------------------------------------------------------
        trainer_cls = TRAINER_REGISTRY[model_cfg["name"]]

        trainer = trainer_cls(
            model=model,
            cfg=model_cfg["training"],
            checkpoint_dir=self.logger.checkpoint_dir(),
        )

        trainer.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            ckpt_path=str(checkpoint_path),
        )

        # IMPORTANT:
        # NNTrainer.fit() must transfer the restored LightningModule
        # weights back to `model` before returning, if the Lightning
        # module and `model` are separate objects.

        # ---------------------------------------------------------
        # 6. Score validation data and fit thresholding
        # ---------------------------------------------------------
        scores = model.score(X_val)

        thresholding_cfg = model_cfg.get("thresholding")

        if thresholding_cfg:
            thresholding = Thresholding(thresholding_cfg)
            thresholding.fit(
                scores=scores,
                y_val=y_val,
            )
            threshold = thresholding.get_threshold()

            threshold_path = self.logger.artifact_path(
                "thresholding/thresholding.pkl"
            )
            thresholding.save(threshold_path)
        else:
            thresholding = None
            threshold = None

        # ---------------------------------------------------------
        # 7. Evaluate and log metrics
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 8. Log training history and diagnostics
        # ---------------------------------------------------------
        history = trainer.history

        if history is not None:
            self.logger.log_training_history(history)

        diagnostics = collect_diagnostics(
            y_true=y_val,
            scores=scores,
            predictions=predictions,
            history=history,
        )

        self.logger.log_diagnostics(diagnostics)
        self.logger.log_candidate_model(
            model,
            metrics,
            model_cfg["name"],
        )

        self.logger.log_tags({"status": "finished"})
        run_status = "FINISHED"

        return {
            "prep": prep,
            "model":
"""








def collect_diagnostics(
    y_true,
    scores,
    predictions,
    history=None,
    attack_families=None,
):
    """Build serializable diagnostic summaries for a binary anomaly detector."""
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    if not (len(y_true) == len(scores) == len(predictions)):
        raise ValueError("y_true, scores, and predictions must have equal lengths.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinite values.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Expected binary labels: 0=normal, 1=anomaly.")
    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("Expected binary predictions: 0=normal, 1=anomaly.")

    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    diagnostics = {
        "data": {
            "n_samples": int(len(y_true)),
            "n_normal": int((y_true == 0).sum()),
            "n_anomaly": int((y_true == 1).sum()),
            "anomaly_rate": float(y_true.mean()) if len(y_true) else 0.0,
        },
        "scores": {
            "min": float(scores.min()) if len(scores) else None,
            "max": float(scores.max()) if len(scores) else None,
            "mean": float(scores.mean()) if len(scores) else None,
            "normal_quantiles": (
                {str(q): float(np.quantile(normal_scores, q))
                 for q in (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)}
                if len(normal_scores) else {}
            ),
            "anomaly_quantiles": (
                {str(q): float(np.quantile(anomaly_scores, q))
                 for q in (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)}
                if len(anomaly_scores) else {}
            ),
            "normal_mean": float(normal_scores.mean()) if len(normal_scores) else None,
            "anomaly_mean": float(anomaly_scores.mean()) if len(anomaly_scores) else None,
        },
        "ranking": {},
        "threshold_metrics": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        },
    }

    if len(np.unique(y_true)) == 2:
        diagnostics["ranking"] = {
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "average_precision": float(average_precision_score(y_true, scores)),
        }
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        diagnostics["pr_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }

    if history is not None:
        train_loss = history.get("train_loss")
        val_loss = history.get("val_loss")
        diagnostics["training"] = {
            "epochs_completed": len(train_loss) if train_loss is not None else 0,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

    if attack_families is not None:
        attack_families = np.asarray(attack_families).reshape(-1)
        if len(attack_families) != len(y_true):
            raise ValueError("attack_families must align with transformed y_true.")
        diagnostics["attack_families"] = {}
        for family in np.unique(attack_families):
            mask = (attack_families == family) & (y_true == 1)
            if not mask.any():
                continue
            diagnostics["attack_families"][str(family)] = {
                "n_samples": int(mask.sum()),
                "mean_score": float(scores[mask].mean()),
                "median_score": float(np.median(scores[mask])),
                "recall": float(predictions[mask].mean()),
            }

    return diagnostics