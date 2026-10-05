import os
from pathlib import Path
import mlflow
import joblib

from .logger import ExperimentLogger

import numpy as np 
import numbers

import pandas as pd


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


#from typing import Optional


from network_anomaly_detection.infra.logging.utils import flatten_dict



from network_anomaly_detection.infra.selection.candidate_registry import CandidateRegistry
from network_anomaly_detection.infra.selection.candidate_manager import CandidateManager


import shutil


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
            min_pr_auc=0.70,
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



    def artifact_pathv0(self, filename):

        run_id = mlflow.active_run().info.run_id

        path = (
            self.artifact_dir
            / "artifacts"
            / run_id
        )

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        return path / filename



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


    # training history
    def log_training_history(
        self,
        history
    ):

        if not history.metrics:
            return

        # ---------- CSV ----------
        df = pd.DataFrame(history.as_dict())

        csv_path = self.artifact_path(
            "history.csv"
        )

        df.to_csv(
            csv_path,
            index=False
        )

        self.log_artifact(
            csv_path,
            artifact_path="training"
        )


        # ---------- One plot per metric ----------
        for metric, values in history.as_dict().items():

            plt.figure()

            plt.plot(
                range(1, len(values) + 1),
                values
            )

            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(metric)

            fig_path = self.artifact_path(
                f"{metric}.png"
            )

            plt.savefig(
                fig_path,
                bbox_inches="tight"
            )

            plt.close()

            self.log_artifact(
                fig_path,
                artifact_path="training"
            )


        # ---------- Combined plot ----------
        plt.figure()

        for metric, values in history.as_dict().items():

            plt.plot(
                range(1, len(values) + 1),
                values,
                label=metric
            )

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training History")
        plt.legend()

        fig_path = self.artifact_path(
            "history.png"
        )

        plt.savefig(
            fig_path,
            bbox_inches="tight"
        )

        plt.close()

        self.log_artifact(
            fig_path,
            artifact_path="training"
        )


    
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


    

    #---------------------------------------------------------#            
    

    # log run
    def log_run(
        self,
        cfg,
        run_type,
        metrics,
        history=None,
        preprocessor=None, # pass already fit preprocessor
        windowing=None, # already fit
        temporal_preprocessor=None, # already fit
        thresholding=None, # already fitted
        wrapper=None,
    ):


        # tags        
        self.log_tags({
            "run_type": run_type,
            "model_type": cfg.get('name'),
        })

        """
        add later
             "trial_number": str(trial.number),
             "optimization_id": study_id,
             dataset hash.....
        """


        # params
        window_size = cfg.get("data", {}).get("windowing", {}).get("size")
        mlflow.log_param("data.windowing.size", window_size)

        self.log_params(
            flatten_dict(cfg.get("prep"))
        )

        self.log_params(
            flatten_dict(cfg.get("models"))
        )

        if cfg.get("training"):
            self.log_params(
                flatten_dict(cfg["training"])
            )


        if cfg.get('thresholding'):
            self.log_params(
                flatten_dict(cfg['thresholding'])    
            )



        # metrics
        self.log_metrics(metrics)

        # training history
        if history and history.metrics:
            self.log_training_history(history)

        # preprocessor artifact
        if preprocessor is not None:
            path = self.artifact_path("preprocessor.pkl")

            preprocessor.save(path)

            self.log_artifact(
                path,
                artifact_path="preprocessing",
            )


        # windowing
        if windowing is not None:
            path = self.artifact_path("windowing.pkl")

            windowing.save(path)

            self.log_artifact(
                path,
                artifact_path="windowing",
            )



        # temporal preprocessor
        if temporal_preprocessor is not None:
            path = self.artifact_path("temporal_preprocessor.pkl")

            temporal_preprocessor.save(path)

            self.log_artifact(
                path,
                artifact_path="temporal_preprocessing",
            )




        # thresholding artifact
        if thresholding is not None:

            path = self.artifact_path(
                "thresholding.pkl"
            )

            thresholding.save(path)

            self.log_artifact(
                path,
                artifact_path="thresholding",
            )



        # model artifact
        if wrapper is not None:

            run = mlflow.active_run()

            run_id = run.info.run_id
            experiment_id = int(run.info.experiment_id)

            model_family = cfg.get("name")
            val_pr_auc = metrics.get("pr_auc")

            retain = (
                val_pr_auc is not None
                and self.candidate_manager.should_retain(
                    experiment_id=experiment_id,
                    model_family=model_family,
                    val_pr_auc=val_pr_auc,
                )
            )

            if retain:

                path = self.artifact_path("model")

                wrapper.save(path)

                self.log_artifact(
                    path,
                    artifact_path="model",
                )

                self.candidate_manager.register_candidate(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    model_family=model_family,
                    val_pr_auc=val_pr_auc,
                    artifact_path="model",
                )







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