from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.preprocessing.pipeline import (
    PreprocessingPipeline,
)
from network_anomaly_detection.preprocessing.components.scalers import create_scaler






class AEPrep(BasePrep):


    def build_pointwise_prep(self, cfg):


        print(cfg)


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



    def build_window_level_prep(self, cfg):
        if not cfg:
            return None

        steps = []

        if cfg.get("delta"):
            steps.append(
                DeltaTransform()
            )

        return PreprocessingPipeline(steps)
    


    def adapt_input(self, X):
        return X.reshape(X.shape[0], -1)