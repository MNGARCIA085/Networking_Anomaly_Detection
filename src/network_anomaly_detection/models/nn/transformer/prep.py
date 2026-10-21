import numpy as np
from network_anomaly_detection.preprocessing.base import BasePrep


class TransformerPrep(BasePrep):
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
One caveat: if your temporal preprocessing includes transformations that depend on preceding 
samples, filtering normal samples before those transformations could change the temporal 
context. Your existing pipeline already has a defined order for preprocessing and sample 
selection, 
so we should preserve that behavior unless we identify a specific reason to change it

analyze more carefully later


IMPORTANT:
- order to choose for preps and data selection

"""