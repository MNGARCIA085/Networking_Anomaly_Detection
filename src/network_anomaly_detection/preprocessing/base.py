import joblib

from abc import ABC, abstractmethod
from network_anomaly_detection.data.windowing import Windowing

from network_anomaly_detection.preprocessing.pipeline import (
    PreprocessingPipeline,
)

from network_anomaly_detection.preprocessing.components.pointwise.scalers import create_scaler
from network_anomaly_detection.preprocessing.components.pointwise.imputation import create_imputer
from network_anomaly_detection.preprocessing.components.pointwise.feature_selection import create_feature_selector
from network_anomaly_detection.preprocessing.components.temporal.delta import create_delta





class BasePrep():
    """Constructs and orchestrates the preprocessing stages."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None

        self.pointwise_prep = None
        self.windowing = None
        self.temporal_prep = None

        self.pointwise_config = None 
        self.windowing_config = None
        self.temporal_config = None



    # Pointwise prep
    def build_pointwise_prep(self, cfg):
        """ It can be override by a model if needed"""
        if not cfg:
            return None

        steps = []
        config = []

        # 1. Imputer
        imputer_cfg = cfg.get("imputation", {})

        if imputer_cfg.get("enabled", False):
            imputer = create_imputer(
                imputer_cfg["name"],
                **imputer_cfg.get("params", {}),
            )
            steps.append(imputer)

            config.append({
                "Imputer": {
                    "name": imputer_cfg["name"],
                    "params": imputer_cfg.get("params", {}),
                }
            })


        # 2. Feature Selection -> Var. threshold (before scaling)
        feat_sel_cfg = cfg.get("feature_selection", {})

        if feat_sel_cfg.get("enabled", False):
            if feat_sel_cfg["name"] == "variance_threshold":
                selector = create_feature_selector(
                    name=feat_sel_cfg["name"],
                    **feat_sel_cfg.get("params", {}),
                )
                steps.append(selector)

                config.append({
                        "name": feat_sel_cfg["name"],
                        "params": feat_sel_cfg.get("params", {}),
                })


        # 3. Scaler
        scaler_cfg = cfg["scaler"]

        scaler = create_scaler(
            scaler_cfg["name"],
            **scaler_cfg.get("params", {}),
        )

        steps.append(scaler)

        config.append({
            "scaler": {
                "name": scaler_cfg["name"],
                "params": scaler_cfg.get("params", {}),
            }
        })


        # 4. Feature Selection -> PCA (after scaling)
        if feat_sel_cfg.get("enabled", False):
            if feat_sel_cfg["name"] == "pca":
                selector = create_feature_selector(
                    name=feat_sel_cfg["name"],
                    **feat_sel_cfg.get("params", {}),
                )
                steps.append(selector)


            config.append({
                "name": feat_sel_cfg["name"],
                "params": feat_sel_cfg.get("params", {}),
            })



        self.pointwise_config = config
        

        return PreprocessingPipeline(steps)



    # Not model-specific, win_size=1 means no actual windowing.
    def build_windowing(self, cfg):

        self.windowing_config = {
            "size": cfg["size"],
            "stride": cfg["stride"],
            "strategy": cfg.get("strategy", "last"),
        }


        return Windowing(
            seq_len=cfg["size"],
            stride=cfg["stride"],
            strategy=cfg.get("strategy", "last"),
        )



    # Temporal prep
    def build_temporal_prep(self, cfg):
        if not cfg:
            return None

        steps = []
        config = []

        # Delta transform
        delta_cfg = cfg.get("delta")

        if delta_cfg and delta_cfg.get("enable", False):
            delta = create_delta(
                delta_cfg["name"],
                **delta_cfg.get("params", {}),
            )
            steps.append(delta)

            config.append({
                "name": delta_cfg["name"],
                "params": delta_cfg.get("params", {}),
            })


        self.temporal_config = config

        return PreprocessingPipeline(steps)



    # Build stages, poinwise -> windowing -> temp. prep.
    def build_preprocessing_stages(self):

        self.pointwise_prep = self.build_pointwise_prep(
            self.cfg.get("prep", {}).get("pointwise")
        )

        self.windowing = self.build_windowing(
            self.cfg.get("prep", {}).get("windowing")
        )

        self.temporal_prep = self.build_temporal_prep(
            self.cfg.get("prep", {}).get("temporal") # temporal_prep
        )

        return (
            self.pointwise_prep,
            self.windowing,
            self.temporal_prep,
        )

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------

    def fit(self, X, y=None):

        self.build_preprocessing_stages()

        # Pointwise preprocessing
        if self.pointwise_prep:
            self.pointwise_prep.fit(X)

            X = self.pointwise_prep.transform(X)

        # Windowing is not fitted.
        if self.windowing:
            if y is not None:
                X, y = self.windowing.transform(X, y)
            else:
                X = self.windowing.transform(X)

        # Temporal preprocessing
        if self.temporal_prep:
            self.temporal_prep.fit(X) # (X, y)

        return self

    # --------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------

    def transform(self, X):

        if self.pointwise_prep:
            X = self.pointwise_prep.transform(X)

        if self.windowing:
            X = self.windowing.transform(X)

        if self.temporal_prep:
            X = self.temporal_prep.transform(X)

        return X

    def transform_with_labels(self, X, y):

        if self.pointwise_prep:
            X = self.pointwise_prep.transform(X)

        if self.windowing:
            X, y = self.windowing.transform(X, y)

        if self.temporal_prep:
            X = self.temporal_prep.transform(X)

        return X, y


    # def. behavior, all data, then , for ex AEs will be trained ONLY with normal data
    def select_training_data(self, X_train, y_train):
        return X_train, y_train


    # --------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)



    #----------- save prep config for logging------------#
    def get_config(self):
        """Return the effective configuration of built preprocessing stages."""

        return {
            "pointwise": self.pointwise_config,
            "windowing": self.windowing_config,
            "temporal": self.temporal_config,
    }



"""
This assumes your PreprocessingPipeline follows the usual scikit-learn convention 
of storing named steps in pipeline.steps


Your custom components, such as Windowing and your delta transformer, might not 
implement get_params().

You have two options:
- Preferred: implement get_params() for your custom components, 
following the scikit-learn convention


"""






"""
# --------------------------------------------------
    # CONVENIENCE METHOD
    # --------------------------------------------------

    def build_prep(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
    ):

        self.fit(
            X_train,
            y_train,
        )

        X_train, y_train = self.transform_with_labels(
            X_train,
            y_train,
        )

        X_val, y_val = self.transform_with_labels(
            X_val,
            y_val,
        )

        return (
            X_train,
            y_train,
            X_val,
            y_val,
        )
"""








