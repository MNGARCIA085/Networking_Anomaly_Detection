import joblib

from abc import ABC, abstractmethod
from network_anomaly_detection.data.windowing import Windowing





from abc import ABC, abstractmethod
import joblib


class BasePrep(ABC):
    """Constructs and orchestrates the preprocessing stages."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None

        self.pointwise_prep = None
        self.windowing = None
        self.temporal_prep = None

    @abstractmethod
    def build_pointwise_prep(self, cfg):
        pass

    @abstractmethod
    def build_temporal_prep(self, cfg):
        pass

    # Not model-specific.
    # win_size=1 means no actual windowing.
    def build_windowing(self, cfg):
        return Windowing(
            cfg.size,
            cfg.stride,
        )

    def build_preprocessing_stages(self):

        self.pointwise_prep = self.build_pointwise_prep(
            self.cfg.get("prep", {}).get("pointwise")
        )

        self.windowing = self.build_windowing(
            self.cfg.get("prep", {}).get("windowing")
        )

        self.temporal_prep = self.build_temporal_prep(
            self.cfg.get("prep", {}).get("temporal_prep")
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

    # --------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)






"""
```python
import joblib
from abc import ABC, abstractmethod

from network_anomaly_detection.data.windowing import Windowing


class BasePrep(ABC):
Constructs and orchestrates preprocessing stages.

    def __init__(self, cfg):
        self.cfg = cfg
        self.pointwise_prep = None
        self.windowing = None
        self.temporal_prep = None

    @abstractmethod
    def build_pointwise_prep(self, cfg):
        pass

    @abstractmethod
    def build_temporal_prep(self, cfg):
        pass

    def build_windowing(self, cfg):
        return Windowing(cfg.size, cfg.stride)

    def build_preprocessing_stages(self):
        self.pointwise_prep = self.build_pointwise_prep(
            self.cfg.get("prep", {}).get("pointwise")
        )

        self.windowing = self.build_windowing(
            self.cfg.get("prep", {}).get("windowing")
        )

        self.temporal_prep = self.build_temporal_prep(
            self.cfg.get("prep", {}).get("temporal_prep")
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

        # Windowing is stateless; labels are aligned if provided.
        if self.windowing:
            if y is not None:
                X, _ = self.windowing.transform(X, y)
            else:
                X = self.windowing.transform(X)

        # Fit on the representation produced by previous stages.
        if self.temporal_prep:
            self.temporal_prep.fit(X)

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

    # --------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
```
Important: this preserves your existing behavior, but assumes Windowing.transform(X, y) returns
 (X_windows, y_windows) and Windowing.transform(X) returns only X_windows. It also assumes 
temporal preprocessing should be fitted on windowed training features.

One caveat: if you later introduce a temporal preprocessor that needs labels to fit, 
this API will need to be extended. For your current design, there are no obvious critical 
bugs in this class based on the code provided.


"""







