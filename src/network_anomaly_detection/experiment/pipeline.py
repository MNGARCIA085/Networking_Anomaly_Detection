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


        print(thresholding_cfg)

        if thresholding_cfg:
            thresholding = Thresholding(thresholding_cfg)


            
            import numpy as np
            print("Scores:", np.min(val_scores), np.max(val_scores))
            print("Unique labels:", np.unique(y_val, return_counts=True))
            print("Anomaly score means:")
            print("  Benign:", val_scores[y_val == 0].mean())
            print("  Anomaly:", val_scores[y_val == 1].mean())
            

            print("Final validation X:", X_val.shape)
            print("Final validation y:", np.unique(y_val, return_counts=True))
            print("Scores:", np.asarray(val_scores).shape)



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




        diagnostics = collect_diagnostics(
            y_true=y_val,
            scores=val_scores,
            predictions=predictions,
            history=trainer.history,
            # attack_families=val_attack_families,  # optional
        )

        self.logger.log_metrics(metrics)
        self.logger.log_diagnostics(diagnostics)  # implement in your logger



        return {
            "prep": prep,
            "model": model,
            "trainer": trainer,
            "thresholding": thresholding,
            "threshold": threshold,
            "metrics": metrics,
            "diagnostics":diagnostics,
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





#---------diagosos, change location later
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def collect_diagnostics(
    y_true,
    scores,
    predictions,
    history=None,
    attack_families=None,
):
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    if not (len(y_true) == len(scores) == len(predictions)):
        raise ValueError(
            "y_true, scores, and predictions must have equal lengths."
        )

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinite values.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Expected binary labels: 0=normal, 1=anomaly.")

    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("Expected binary predictions: 0=normal, 1=anomaly.")

    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    diagnostics = {
        "data": {
            "n_samples": int(len(y_true)),
            "n_normal": int((y_true == 0).sum()),
            "n_anomaly": int((y_true == 1).sum()),
            "anomaly_rate": float(y_true.mean()),
        },
        "scores": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "normal_quantiles": (
                {
                    str(q): float(np.quantile(normal_scores, q))
                    for q in (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
                }
                if len(normal_scores) else {}
            ),
            "anomaly_quantiles": (
                {
                    str(q): float(np.quantile(anomaly_scores, q))
                    for q in (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
                }
                if len(anomaly_scores) else {}
            ),
            "normal_mean": (
                float(normal_scores.mean()) if len(normal_scores) else None
            ),
            "anomaly_mean": (
                float(anomaly_scores.mean()) if len(anomaly_scores) else None
            ),
        },
        "ranking": {},
        "threshold_metrics": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": (
                float(2 * tp / (2 * tp + fp + fn))
                if 2 * tp + fp + fn else 0.0
            ),
            "specificity": (
                float(tn / (tn + fp)) if tn + fp else 0.0
            ),
        },
    }

    # Ranking metrics require both classes.
    if len(np.unique(y_true)) == 2:
        diagnostics["ranking"] = {
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "average_precision": float(
                average_precision_score(y_true, scores)
            ),
        }

        precision, recall, thresholds = precision_recall_curve(
            y_true, scores
        )
        diagnostics["pr_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }

    if history is not None:
        train_loss = history.get("train_loss")
        val_loss = history.get("val_loss")

        diagnostics["training"] = {
            "epochs_completed": len(train_loss),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

    if attack_families is not None:
        attack_families = np.asarray(attack_families).reshape(-1)

        if len(attack_families) != len(y_true):
            raise ValueError(
                "attack_families must align with transformed y_true."
            )

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