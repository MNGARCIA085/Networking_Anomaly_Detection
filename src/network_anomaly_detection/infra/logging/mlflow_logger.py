import os
from pathlib import Path
import mlflow
import joblib
import shutil
import json

from sklearn.metrics import confusion_matrix, precision_recall_curve

import pandas as pd
import numpy as np 
import numbers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from .logger import ExperimentLogger

from network_anomaly_detection.infra.logging.utils import flatten_dict
from network_anomaly_detection.infra.selection.candidate_registry import CandidateRegistry
from network_anomaly_detection.infra.selection.candidate_manager import CandidateManager






"""
rm -rf mlruns/
rm -rf mlartifacts/      
rm -f mlflow.db 
"""


"""
isolate models

model_path = self.artifact_path(
    path
)

"""




class MLFlowLogger(ExperimentLogger):

    def __init__(
        self,
        exp_name='Anomaly_Detection',
        tracking_db="mlflow.db",
        artifact_dir="mlruns"
    ):


        self.tracking_db = Path(tracking_db)
        self.artifact_dir = Path(artifact_dir)
        self.exp_name = exp_name
        self._init_mlflow()

        self.candidate_registry = CandidateRegistry(
            f"sqlite:///{self.tracking_db}"
        )

        self.candidate_manager = CandidateManager(
            registry=self.candidate_registry,
            #mlflow_dir=self.artifact_dir,
            logger = self,
            candidate_pool_size=5,
            min_pr_auc=0.50,
            max_candidates_per_model=2,
        )
        # later -> values from YAML


    def _init_mlflow(self):

        mlflow.set_tracking_uri(
            f"sqlite:///{self.tracking_db}"
        )

        os.makedirs(
            self.artifact_dir,
            exist_ok=True
        )

        mlflow.set_experiment(
            self.exp_name
        )


    def start_run(
        self,
        run_name=None
    ):
        return mlflow.start_run(
            run_name=run_name
        )


    def end_run(self):
        mlflow.end_run()



    def log_tags(self, tags: dict):
        if not tags:
            return

        mlflow.set_tags(tags)


    def log_params(
        self,
        params
    ):
        mlflow.log_params(
            params
        )


    def log_metrics(self, metrics):

        clean_metrics = {}

        for k, v in metrics.items():

            if isinstance(v, np.generic):
                v = v.item()

            if isinstance(v, numbers.Number):
                clean_metrics[k] = float(v)

        mlflow.log_metrics(clean_metrics)



    def log_artifact(
        self,
        path,
        artifact_path=None
    ):

        path = Path(path)

        if path.is_dir():
            mlflow.log_artifacts(
                str(path),
                artifact_path=artifact_path
            )

        else:
            mlflow.log_artifact(
                str(path),
                artifact_path=artifact_path
            )


    def log_model(
        self,
        model,
        path
    ):

        model_path = (
            self.artifact_dir / path
        )

        joblib.dump(
            model,
            model_path
        )

        mlflow.log_artifact(
            str(model_path)
        )




    
    def log_training_history(self, history):
        if not history:
            return

        training_dir = self.run_artifact_dir() / "training"
        training_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        df = pd.DataFrame(history)
        csv_path = training_dir / "history.csv"
        df.to_csv(csv_path, index=False)
        self.log_artifact(csv_path, artifact_path="training")

        # One plot per metric
        for metric, values in history.items():
            if not values:
                continue

            fig, ax = plt.subplots()
            ax.plot(range(1, len(values) + 1), values)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(metric)

            fig_path = training_dir / f"{metric}.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            self.log_artifact(fig_path, artifact_path="training")

        # Combined plot
        metrics = {
            metric: values
            for metric, values in history.items()
            if values
        }

        if metrics:
            fig, ax = plt.subplots()

            for metric, values in metrics.items():
                ax.plot(
                    range(1, len(values) + 1),
                    values,
                    label=metric,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_title("Training History")
            ax.legend()

            fig_path = training_dir / "history.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            self.log_artifact(fig_path, artifact_path="training")


    
    def delete_artifact(self, run_id, artifact_path):
        # MLflow local artifact structure:
        #
        # mlruns/
        #   <experiment_id>/
        #       <run_id>/
        #           artifacts/
        #

        client = mlflow.tracking.MlflowClient()

        run = client.get_run(run_id)
        experiment_id = run.info.experiment_id

        target = (
            self.artifact_dir
            / str(experiment_id)
            / run_id
            / "artifacts"
            / artifact_path
        )

        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()




    

    #--------------------------------------------------------#
    



    def run_artifact_dir(self):
        """Return the local artifact directory for the active MLflow run."""

        run = mlflow.active_run()

        if run is None:
            raise RuntimeError(
                "No active MLflow run."
            )

        run_id = run.info.run_id

        path = (
            self.artifact_dir
            / str(run.info.experiment_id)
            / run_id
            / "artifacts"
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path


    def artifact_path(
        self,
        filename,
        artifact_dir=None,
    ):
        """Return a local path inside the active run's artifacts."""

        base = (
            artifact_dir
            if artifact_dir is not None
            else self.run_artifact_dir()
        )

        path = base / filename

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path



    def checkpoint_dir(self):
        return self.run_artifact_dir() / "checkpoints"


    # real ones for checkpopints
    def run_artifact_dir(self):
        run = mlflow.active_run()

        if run is None:
            raise RuntimeError(
                "No active MLflow run."
            )

        path = (
            self.artifact_dir
            / str(run.info.experiment_id)
            / run.info.run_id
            / "artifacts"
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path


    def run_artifact_dir_by_id(self, run_id):

        client = mlflow.tracking.MlflowClient()

        run = client.get_run(run_id)

        path = Path(
            run.info.artifact_uri.replace(
                "file://",
                "",
            )
        )

        return path



    # little more ...
    def get_run_artifact_path(self, run_id, artifact_path):
        client = mlflow.tracking.MlflowClient()

        run = client.get_run(run_id)

        artifact_uri = run.info.artifact_uri

        if artifact_uri.startswith("file://"):
            artifact_uri = artifact_uri[7:]

        return Path(artifact_uri) / artifact_path




    def get_checkpoint_path(self, run_id, checkpoint=None):
        checkpoint_dir = self.get_run_artifact_path(
            run_id,
            "checkpoints",
        )

        if checkpoint is None:
            checkpoints = sorted(checkpoint_dir.glob("*.pt"))

            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoints found for run {run_id}."
                )

            return checkpoints[-1]

        path = checkpoint_dir / checkpoint

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        return path



    #----------------save models--------------------#
    def log_candidate_model(
        self,
        model,
        metrics,
        model_family,
    ):
        run = mlflow.active_run()

        if run is None:
            raise RuntimeError("No active MLflow run.")

        val_pr_auc = metrics.get("pr_auc")


        if val_pr_auc is None:
            return False

        
        retain = self.candidate_manager.should_retain(
            experiment_id=int(run.info.experiment_id),
            model_family=model_family,
            val_pr_auc=float(val_pr_auc),
        )

        #if not retain:
        #    return False
        

        path = self.artifact_path("model")

        model.save(path)


        """
        self.log_artifact(
            path,
            artifact_path="model",
        )
        """


        self.candidate_manager.register_candidate(
            experiment_id=int(run.info.experiment_id),
            run_id=run.info.run_id,
            model_family=model_family,
            val_pr_auc=float(val_pr_auc),
            artifact_path="model",
        )

        return True




    #-------------------------------------------------------#
    def log_diagnostics(self, diagnostics):
        # Scalar diagnostics -> MLflow metrics
        metrics = {}

        def collect_scalars(data, prefix=""):
            for key, value in data.items():
                name = f"{prefix}{key}"

                if isinstance(value, dict):
                    collect_scalars(value, prefix=f"{name}_")
                elif isinstance(value, (int, float, np.number)):
                    if np.isfinite(value):
                        metrics[name] = float(value)

        collect_scalars(diagnostics)

        if metrics:
            self.log_metrics(metrics)

        # Full diagnostics -> JSON artifact
        path = self.artifact_path("diagnostics/diagnostics.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, allow_nan=False)

        return path


    #---------------------------------------------------------#    
    def log_config(self, config: dict):
        """Log the resolved experiment configuration as a JSON artifact."""
        path = self.artifact_path("config.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                config,
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            )

        self.log_artifact(path)
        return path


    #----------------- save prep (generalioze later)-------------#
    def save_prep(self, prep):
        # Keep this artifact path consistent with resume().
        prep_path = self.artifact_path(
            "prep.pkl", #joblib
            artifact_dir=self.run_artifact_dir() / "preprocessing",
        )
        prep.save(prep_path)
        self.log_artifact(prep_path, artifact_path="preprocessing")



    #------------more diagnosis data---------------#
    def log_evaluation_diagnostics(self, y_true, scores, predictions):
        
        y_true = np.asarray(y_true).ravel()
        scores = np.asarray(scores).ravel()
        predictions = np.asarray(predictions).ravel()

        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]

        quantiles = (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

        diagnostics = {
            "score_distribution": {
                "min": float(scores.min()) if len(scores) else None,
                "max": float(scores.max()) if len(scores) else None,
                "mean": float(scores.mean()) if len(scores) else None,
                "normal_mean": (
                    float(normal_scores.mean()) if len(normal_scores) else None
                ),
                "anomaly_mean": (
                    float(anomaly_scores.mean()) if len(anomaly_scores) else None
                ),
                "normal_quantiles": {
                    str(q): float(np.quantile(normal_scores, q))
                    for q in quantiles
                } if len(normal_scores) else {},
                "anomaly_quantiles": {
                    str(q): float(np.quantile(anomaly_scores, q))
                    for q in quantiles
                } if len(anomaly_scores) else {},
            },
        }

        path = self.artifact_path("diagnostics/diagnostics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, allow_nan=False)

        self.log_artifact(path, artifact_path="diagnostics")

        
        # Confusion matrix, PR curve, score distributions.

        if not (len(y_true) == len(scores) == len(predictions)):
            raise ValueError("Inputs must have equal lengths.")

        def save_plot(fig, filename):
            path = self.artifact_path(f"diagnostics/{filename}")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            self.log_artifact(path, artifact_path="diagnostics")

        # 1. Confusion matrix
        cm = confusion_matrix(y_true, predictions, labels=[0, 1])

        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            xlabel="Predicted label",
            ylabel="True label",
            title="Confusion Matrix",
        )

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        save_plot(fig, "confusion_matrix.png")

        # 2. Precision-recall curve
        if len(np.unique(y_true)) == 2:
            precision, recall, _ = precision_recall_curve(y_true, scores)

            fig, ax = plt.subplots()
            ax.plot(recall, precision, label="Precision–Recall")
            ax.axhline(
                y=y_true.mean(),
                linestyle="--",
                label="Positive-class prevalence",
            )
            ax.set(
                xlabel="Recall",
                ylabel="Precision",
                title="Precision–Recall Curve",
            )
            ax.legend()
            save_plot(fig, "pr_curve.png")

        # 3. Score separation
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]

        fig, ax = plt.subplots()
        ax.hist(normal_scores, bins=40, density=True, alpha=0.6,
                label="Normal")
        ax.hist(anomaly_scores, bins=40, density=True, alpha=0.6,
                label="Anomaly")
        ax.set(
            xlabel="Anomaly score",
            ylabel="Density",
            title="Score Distributions by Class",
        )
        ax.legend()
        save_plot(fig, "score_separation.png")




    #------log val scores----------#
    def log_validation_scores(self, y_true, scores, predictions):

        df = pd.DataFrame({
            "y_true": np.asarray(y_true).ravel(),
            "score": np.asarray(scores).ravel(),
            "prediction": np.asarray(predictions).ravel(),
        })

        path = self.artifact_path("diagnostics/validation_scores.csv")
        df.to_csv(path, index=False)
        self.log_artifact(path, artifact_path="diagnostics")

        return path


    


      







"""
train
  ↓
validation PR-AUC
  ↓
should_retain()?
  ├── NO → MLflow metrics only
  │
  └── YES
       ↓
   save model
       ↓
   MLflow artifact
       ↓
   register candidate
       ↓
   pool > N?
       ├── NO
       └── YES → delete weakest artifact
"""



"""
Ownership
MLFlowLogger
    │
    ├── CandidateRegistry
    │
    └── CandidateManager
             │
             ├── registry
             └── logger ────┐
                            │
                       MLFlowLogger

"""




"""
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000


rm -rf mlruns/
rm -rf mlartifacts/      
rm -f mlflow.db  


# if it exists
        # if using SQLite

if I save outside
rm -rf artifacts/
rm -rf checkpoints/
rm -rf saved_models/


"""


"""
load threshold later
The important part is that thresholding.pkl contains the fitted strategy:

thresholding.strategy.threshold

will contain, for example:

0.13742

So at inference you simply load it:

thresholding = Thresholding.load(
    "thresholding.pkl"
)


predictions = thresholding.predict(
    scores
)
"""