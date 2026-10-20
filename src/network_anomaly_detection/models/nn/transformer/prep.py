import numpy as np

from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.preprocessing.pipeline import (
    PreprocessingPipeline,
)

from network_anomaly_detection.preprocessing.components.pointwise.scalers import create_scaler
from network_anomaly_detection.preprocessing.components.pointwise.imputation import create_imputer
from network_anomaly_detection.preprocessing.components.temporal.delta import create_delta


class TransformerPrep(BasePrep):


    def build_pointwise_prep(self, cfg):
        if not cfg:
            return None

        steps = []


        # Imputer
        imputer_cfg = cfg.get("imputation", {})

        if imputer_cfg.get("enabled", False):
            imputer = create_imputer(
                imputer_cfg["name"],
                **imputer_cfg.get("params", {}),
            )
            steps.append(imputer)


        # scaler
        scaler_cfg = cfg["scaler"]

        scaler = create_scaler(
            scaler_cfg["name"],
            **scaler_cfg.get("params", {}),
        )

        steps.append(scaler)

        return PreprocessingPipeline(steps)



    def build_temporal_prep(self, cfg):
        if not cfg:
            return None

        steps = []

        # Delta transform
        delta_cfg = cfg.get("delta")

        if delta_cfg and delta_cfg.get("enable", False):
            delta = create_delta(
                delta_cfg["name"],
                **delta_cfg.get("params", {}),
            )
            steps.append(delta)

        return PreprocessingPipeline(steps)
    


    # select only normal data
    def select_training_data(self, X_train, y_train):
        mask = np.asarray(y_train).reshape(-1) == 0

        if hasattr(X_train, "loc"):
            X_train = X_train.loc[mask]
        else:
            X_train = X_train[mask]

        if hasattr(y_train, "loc"):
            y_train = y_train.loc[mask]
        else:
            y_train = y_train[mask]

        return X_train, y_train





"""
One caveat: if your temporal preprocessing includes transformations that depend on preceding 
samples, filtering normal samples before those transformations could change the temporal 
context. Your existing pipeline already has a defined order for preprocessing and sample 
selection, 
so we should preserve that behavior unless we identify a specific reason to change it

analyze more carefully later


IMPORTANT:
- order to choose for preps and data selection

"""