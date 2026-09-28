import joblib

from abc import ABC, abstractmethod
from network_anomaly_detection.data.windowing import Windowing



class BasePrep(ABC):
    """ constructs and orchestrates the preprocessing stages """

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None


    @abstractmethod
    def build_pointwise_prep(self, cfg):
        pass

    @abstractmethod
    def build_temporal_prep(self, cfg):
        pass

    # actually is not per class; and win_size=1 equals no windowing
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
        

        return (
            self.pointwise_prep,
            self.windowing,
            self.temporal_prep,
        )



    def build_prep(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        pointwise_prep, windowing, temporal_prep = (
            self.build_preprocessing_stages()
        )



        print(X_train.shape)

        # Pointwise
        if pointwise_prep:
            X_train = pointwise_prep.fit_transform(X_train)
            X_val = pointwise_prep.transform(X_val)


        print(X_train.shape)



        # Windowing
        if windowing:
            X_train, y_train = windowing.transform(
                X_train,
                y_train,
            )

            X_val, y_val = windowing.transform(
                X_val,
                y_val,
            )


        print(X_train.shape)

        # temporal prep
        if temporal_prep:
            X_train = temporal_prep.fit_transform(X_train)
            X_val = temporal_prep.transform(X_val)


        # adapt input here or in the model??????

        return (
            X_train,
            y_train,
            X_val,
            y_val,
        )


    # for inference later
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



    #------SAVE AND LOAD PIPELINE---------#
    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)






