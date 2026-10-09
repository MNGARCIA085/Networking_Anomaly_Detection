import numpy as np

from sklearn.ensemble import IsolationForest

from network_anomaly_detection.models.base_model import BaseModel
from network_anomaly_detection.models.ml.isoforest.schemas import IsoForestConfig


from network_anomaly_detection.models.persistence.sklearn import save_sklearn_model, load_sklearn_model

from pathlib import Path
import joblib



class IsoForestModel(BaseModel):

    def __init__(self, cfg: dict, input_shape=None):
        self.config = cfg

        self.model = IsolationForest(
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
            max_samples=cfg["max_samples"],
            max_features=cfg["max_features"],
            bootstrap=cfg["bootstrap"],
            random_state=cfg["random_state"],
        )

    def adapt_input(self, X):
        # (N, W, F) -> (N, W * F)
        return X.reshape(X.shape[0], -1)

    def score(self, X):
        # sklearn: higher = more normal
        # project convention: higher = more anomalous
        return -self.model.decision_function(X)

    def predict(self, X, threshold=None):
        if threshold is not None:
            return (self.score(X) >= threshold).astype(int)

        # sklearn: 1 = inlier, -1 = outlier
        return (self.model.predict(X) == -1).astype(int)

    def fit(self, X):
        self.model.fit(X)
        return self





    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_sklearn_model(self.model, path)

        joblib.dump(
            {"config": self.config},
            path / "wrapper_config.pkl",
        )


    @classmethod
    def load(cls, path):
        path = Path(path)

        wrapper_cfg = joblib.load(path / "wrapper_config.pkl")

        instance = cls.__new__(cls)
        instance.config = wrapper_cfg["config"]
        instance.model = load_sklearn_model(path)

        return instance



    """
    # save & load
    def save(self, path):

        save_sklearn_model(
            self.model,
            path
        )


    @classmethod
    def load(cls, path):

        model = load_sklearn_model(
            path
        )

        return cls(
            model=model
        )
    """






class IsoForestModelv0(BaseModel):

    def __init__(self, cfg: IsoForestConfig, input_shape=None):

        self.config = cfg

        self.model = IsolationForest(
            n_estimators=cfg.n_estimators,
            contamination=cfg.contamination,
            max_samples=cfg.max_samples,
            max_features=cfg.max_features,
            bootstrap=cfg.bootstrap,
            random_state=cfg.random_state,
        )

    def adapt_input(self, X):

        # (N, W, F) -> (N, W * F)
        return X.reshape(X.shape[0], -1)

    def score(self, X):

        # sklearn: higher = more normal
        # project convention: higher = more anomalous
        return -self.model.decision_function(X)

    def predict(self, X, threshold=None):

        # sklearn:
        #  1 = inlier
        # -1 = outlier
        return (self.model.predict(X) == -1).astype(int)



    # for the trainer; discuss real location later
    def fit(self, X):
        self.model.fit(X)
        return self



    """
    # later, for unform inference
    @property
    def input_dim(self):
        return self.model.n_features_in_ # number of features stored by the fitted IsolationForest



    # save & load
    def save(self, path):

        save_sklearn_model(
            self.model,
            path
        )


    @classmethod
    def load(cls, path):

        model = load_sklearn_model(
            path
        )

        return cls(
            model=model
        )
    """









    

