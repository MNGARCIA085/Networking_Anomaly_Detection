import numpy as np

from network_anomaly_detection.preprocessing.base import BasePrep
from network_anomaly_detection.preprocessing.pipeline import (
    PreprocessingPipeline,
)

from network_anomaly_detection.preprocessing.components.pointwise.scalers import create_scaler
from network_anomaly_detection.preprocessing.components.pointwise.imputation import create_imputer
from network_anomaly_detection.preprocessing.components.temporal.delta import create_delta
from network_anomaly_detection.preprocessing.components.pointwise.feature_selection import create_feature_selector



"""
var. threshodl bfore scaling; PCA after


"""






class AEPrep(BasePrep):


    def build_pointwise_prep(self, cfg):
        if not cfg:
            return None

        steps = []

        # 1. Imputer
        imputer_cfg = cfg.get("imputation", {})

        if imputer_cfg.get("enabled", False):
            imputer = create_imputer(
                imputer_cfg["name"],
                **imputer_cfg.get("params", {}),
            )
            steps.append(imputer)


        # 2. Feature Selection -> Var. threshold (before scaling)
        feat_sel_cfg = cfg.get("feature_selection", {})

        if feat_sel_cfg.get("enabled", False):
            if feat_sel_cfg["name"] == "variance_threshold":
                selector = create_feature_selector(
                    name=feat_sel_cfg["name"],
                    **feat_sel_cfg.get("params", {}),
                )
                steps.append(selector)


        # 3. Scaler
        scaler_cfg = cfg["scaler"]

        scaler = create_scaler(
            scaler_cfg["name"],
            **scaler_cfg.get("params", {}),
        )

        steps.append(scaler)



        # 4. Feature Selection -> PCA (after scaling)
        if feat_sel_cfg.get("enabled", False):
            if feat_sel_cfg["name"] == "pca":
                selector = create_feature_selector(
                    name=feat_sel_cfg["name"],
                    **feat_sel_cfg.get("params", {}),
                )
                steps.append(selector)
        

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
    def select_training_data(self, X_train, y_train):
        mask = y_train == 0
        X_train = X_train.loc[mask]
        y_train = y_train.loc[mask]

        return X_train, y_train
    """
