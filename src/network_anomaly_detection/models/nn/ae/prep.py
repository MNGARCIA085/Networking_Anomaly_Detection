from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.preprocessing.pipeline import (
    PreprocessingPipeline,
)

from network_anomaly_detection.preprocessing.components.pointwise.scalers import create_scaler
from network_anomaly_detection.preprocessing.components.temporal.delta import create_delta





class AEPrep(BasePrep):


    def build_pointwise_prep(self, cfg):
        if not cfg:
            return None

        steps = []

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
    


    def adapt_input(self, X):
        return X.reshape(X.shape[0], -1)