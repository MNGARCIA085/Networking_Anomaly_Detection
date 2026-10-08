import copy
import optuna

from network_anomaly_detection.experiment.experiment import Experiment
from .registry import TUNING_REGISTRY


class Tuner:

    def __init__(self, cfg, evaluator, logger):

        self.cfg = cfg
        self.evaluator = evaluator
        self.logger = logger

        self.tun_cfg = cfg.model_type.tuning

        self.sampler = TUNING_REGISTRY[
            cfg.model_type.name
        ](
            #tuning_cfg=self.tun_cfg,
            base_cfg=cfg,
        )


    def run(
        self,
        prep,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=None,
        n_jobs=None,
    ):

        # Use YAML defaults unless explicitly overridden
        if n_trials is None:
            n_trials = self.tun_cfg.n_trials

        if n_jobs is None:
            n_jobs = self.tun_cfg.n_jobs

        metric = self.tun_cfg.metric
        direction = self.tun_cfg.direction

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
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                run_name=f"trial_{trial.number}",
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

        return study.best_trial.user_attrs[
            "config"
        ]