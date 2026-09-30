from network_anomaly_detection.training.base_trainer import BaseTrainer


class IsoForestTrainer(BaseTrainer):

    def fit(
        self,
        X,
        y=None,
        X_val=None,
        y_val=None,
    ):
        self.model.fit(X)
        return self