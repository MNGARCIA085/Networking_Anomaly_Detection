from pathlib import Path

import joblib
import numpy as np

from network_anomaly_detection.models.base_model import BaseModel
from network_anomaly_detection.models.nn.ae.model import AEModel
from network_anomaly_detection.models.ml.isoforest.model import IsoForestModel


class AEIsolationForestModel(BaseModel):

    def __init__(self, cfg: dict, input_shape):
        self.config = cfg
        self.input_shape = tuple(input_shape)

        self.ae = AEModel(
            cfg=cfg["ae"],
            input_shape=input_shape,
        )

        self.isoforest = IsoForestModel(
            cfg=cfg["isoforest"],
        )

    def adapt_input(self, X):
        """Adapt canonical input using the AE's input adaptation."""
        return self.ae.adapt_input(X)

    def encode(self, X):
        """Encode already-adapted input into the AE latent space."""
        return self.ae.encode(X)

    def fit_detector(self, X):
        """Fit Isolation Forest on AE latent representations."""
        representations = self.encode(X)
        self.isoforest.fit(representations)
        return self

    def score(self, X):
        """Return anomaly scores from Isolation Forest in latent space."""
        representations = self.encode(X)
        return self.isoforest.score(representations)

    def predict(self, X, threshold):
        return (self.score(X) >= threshold).astype(int)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.ae.save(path / "ae")
        self.isoforest.save(path / "isoforest")

        joblib.dump(
            {
                "config": self.config,
                "input_shape": self.input_shape,
            },
            path / "wrapper_config.pkl",
        )

    @classmethod
    def load(cls, path):
        path = Path(path)

        wrapper_cfg = joblib.load(path / "wrapper_config.pkl")

        instance = cls.__new__(cls)
        instance.config = wrapper_cfg["config"]
        instance.input_shape = tuple(wrapper_cfg["input_shape"])

        instance.ae = AEModel.load(path / "ae")
        instance.isoforest = IsoForestModel.load(path / "isoforest")

        return instance