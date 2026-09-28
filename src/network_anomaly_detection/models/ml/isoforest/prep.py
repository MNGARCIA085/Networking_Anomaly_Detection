from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.preprocessing.pipeline import (
    PreprocessingPipeline,
)

from network_anomaly_detection.preprocessing.components.pointwise.dimensionality_reduction import create_dimensionality_reducer
from network_anomaly_detection.preprocessing.components.pointwise.feature_selection import create_feature_selector
from network_anomaly_detection.preprocessing.components.pointwise.transforms import create_transform
from network_anomaly_detection.preprocessing.components.pointwise.scalers import create_scaler





class IsoPrep(BasePrep):


    def build_pointwise_prep(self, cfg):
        if not cfg:
            return None

        steps = []

        # Feature Selection
        feat_sel_cfg = cfg.get("feature_selection")

        if feat_sel_cfg and feat_sel_cfg.get("enable", False):
            feature_selector = create_feature_selector(
                feat_sel_cfg["name"],
                **feat_sel_cfg.get("params", {}),
            )
            steps.append(feature_selector)


    
        # Transform
        transform_cfg = cfg.get("transform")

        if transform_cfg and transform_cfg.get("enable", False):
            transform = create_transform(
                transform_cfg["name"],
                **transform_cfg.get("params", {}),
            )
            steps.append(transform)


        # Scaler
        scaler_cfg = cfg["scaler"]

        if scaler_cfg and scaler_cfg.get("enable", False):
            scaler = create_scaler(
                scaler_cfg["name"],
                **scaler_cfg.get("params", {}),
            )

            steps.append(scaler)


        # Dimensionality Reduction
        dim_red_cfg = cfg.get("dim_reduction")


        if dim_red_cfg and dim_red_cfg.get("enable", False):
            dim_reducer = create_dimensionality_reducer(
                dim_red_cfg["name"],
                **dim_red_cfg.get("params", {}),
            )
            steps.append(dim_reducer)



        return PreprocessingPipeline(steps)




    def build_temporal_prep(self, cfg):
        if not cfg:
            return None

        # see later if what it makes sense to try

        steps = []

        return PreprocessingPipeline(steps)
    


    def adapt_input(self, X):
        return X.reshape(X.shape[0], -1)