import copy
import optuna
from copy import deepcopy

from network_anomaly_detection.experiment.experiment import Experiment
from network_anomaly_detection.preprocessing.registry import PREP_REGISTRY
from .registry import TUNING_REGISTRY


class Tuner:

    def __init__(self, cfg, evaluator, logger):

        self.cfg = cfg
        self.evaluator = evaluator
        self.logger = logger

        self.tun_cfg = cfg["model_type"]["tuning"]

        self.sampler = TUNING_REGISTRY[
            cfg["model_type"]["name"]
        ](
            base_cfg=cfg,
            tuning_cfg=self.tun_cfg,
        )


    def run(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        parent_run_id=None,
        n_trials=None,
        n_jobs=None,
    ):

        # Use YAML defaults unless explicitly overridden
        if n_trials is None:
            n_trials = self.tun_cfg["n_trials"]

        if n_jobs is None:
            n_jobs = self.tun_cfg["n_jobs"]

        metric = self.tun_cfg["metric"]
        direction = self.tun_cfg["direction"]

        def objective(trial):

            # --------------------------------------------------
            # Model-specific sampling
            # --------------------------------------------------

            trial_cfg = self.sampler.sample(
                trial
            )

            trial.set_user_attr(
                "config",
                trial_cfg,
            )

            # Create a fresh prep for this trial's configuration.
            prep_cls = PREP_REGISTRY[trial_cfg["model_type"]["name"]]
            prep = prep_cls(trial_cfg["model_type"])

            # Apply model-specific selection to training data.
            X_trial_train, y_trial_train = prep.select_training_data(
                X_train.copy(),
                y_train.copy(),
            )

            # Fit preprocessing exclusively on this trial's training data.
            prep.fit(X_trial_train, y_trial_train)

            # --------------------------------------------------
            # One independent Experiment per trial
            # --------------------------------------------------

            experiment = Experiment(
                cfg=trial_cfg,
                logger=self.logger,
            )

            result = experiment.run(
                prep=prep,
                evaluator=self.evaluator,
                X_train=X_trial_train,
                y_train=y_trial_train,
                X_val=X_val,
                y_val=y_val,
                nested=True,
                run_name=f"trial_{trial.number}",
                metadata={
                    "mlflow.parentRunId": parent_run_id,
                    "run_type": "tuning_trial",
                    "trial_number": str(trial.number),
                    "tuning_metric": metric,
                },
            )

            # --------------------------------------------------
            # Objective
            # --------------------------------------------------

            score = result["metrics"][metric]

            return score

        # ------------------------------------------------------
        # Optuna study
        # ------------------------------------------------------

        study = optuna.create_study(
            direction=direction,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

        return study



    
    @staticmethod
    def get_best_config(study):
        cfg = deepcopy(study.best_trial.user_attrs["config"])
        # remove unnecessary keys
        cfg["model_type"].pop("tuning", None)
        cfg.pop("paths", None)
        cfg.pop("tracking", None)
        cfg.pop("experiment_name", None)
        return cfg

    