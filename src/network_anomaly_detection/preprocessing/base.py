from abc import ABC, abstractmethod


from network_anomaly_detection.data.windowing import Windowing


import joblib



class BasePrep(ABC):

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None


    @abstractmethod
    def build_pointwise_prep(self, cfg):
        pass

    @abstractmethod
    def build_window_level_prep(self, cfg):
        pass

    # actually is not per class; and win_size=1 equals no windowing
    def build_windowing(self, cfg):
        return Windowing(2) # later from config


    # is not exactly a pipeline
    def build_pipeline(self):



        self.pointwise_prep = self.build_pointwise_prep(
            self.cfg.get("prep", None)
        )

        self.windowing = self.build_windowing(
            self.cfg.get("data", {}).get("windowing")
        )

        self.window_level_prep = self.build_window_level_prep(
            self.cfg.get("prep", {}).get("window_level")
        )


        """ later this specific
        self.pointwise_prep = self.build_pointwise_prep(
            self.cfg.get("prep", {}).get("pointwise")
        )

        self.windowing = self.build_windowing(
            self.cfg.get("data", {}).get("windowing")
        )

        self.window_level_prep = self.build_window_level_prep(
            self.cfg.get("prep", {}).get("window_level")
        )
        """

        return (
            self.pointwise_prep,
            self.windowing,
            self.window_level_prep,
        )



    def build_prep(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        pointwise_prep, windowing, window_level_prep = (
            self.build_pipeline()
        )

        # Pointwise
        if pointwise_prep:
            X_train = pointwise_prep.fit_transform(X_train)
            X_val = pointwise_prep.transform(X_val)

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

        # Window-level
        if window_level_prep:
            X_train = window_level_prep.fit_transform(X_train)
            X_val = window_level_prep.transform(X_val)


        # adapt input here or in the model??????

        return (
            X_train,
            y_train,
            X_val,
            y_val,
        )



    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)



    #-------------------------------
    def transform(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        if self.pointwise_prep:
            X_train = self.pointwise_prep.transform(X_train)
            X_val = self.pointwise_prep.transform(X_val)

        if self.windowing:
            X_train, y_train = self.windowing.transform(
                X_train,
                y_train,
            )

            X_val, y_val = self.windowing.transform(
                X_val,
                y_val,
            )

        if self.window_level_prep:
            X_train = self.window_level_prep.transform(X_train)
            X_val = self.window_level_prep.transform(X_val)

        return (
            X_train,
            y_train,
            X_val,
            y_val,
        )






    """
    @abstractmethod
    def build_pipeline(self):
        raise NotImplementedError


    def fit(self, X):
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X)

        return self


    def transform(self, X, y=None):
        return self.pipeline.transform(X, y)


    def get_artifacts(self):
        return {
            "pipeline": self.pipeline,
        }
    """