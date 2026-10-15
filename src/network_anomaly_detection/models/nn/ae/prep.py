from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.preprocessing.pipeline import (
    PreprocessingPipeline,
)

from network_anomaly_detection.preprocessing.components.pointwise.scalers import create_scaler
from network_anomaly_detection.preprocessing.components.pointwise.imputation import create_imputer
from network_anomaly_detection.preprocessing.components.temporal.delta import create_delta





class AEPrep(BasePrep):


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
        mask = y_train == 0
        X_train = X_train.loc[mask]
        y_train = y_train.loc[mask]

        return X_train, y_train

