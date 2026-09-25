# data/data_module.py

from pathlib import Path

import pandas as pd


class DataModule:

    def __init__(
        self,
        train_path: str | Path,
        val_path: str | Path,
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)


    def load(self):
        train = pd.read_csv(self.train_path)
        val = pd.read_csv(self.val_path)

        X_train, y_train = self._split_features_labels(train)
        X_val, y_val = self._split_features_labels(val)


        return (
            X_train,
            y_train,
            X_val,
            y_val,
        )

    # load test in other method...

    @staticmethod
    def _split_features_labels(df: pd.DataFrame):
        y = df[" Label"] # corect later!!!!!!
        X = df.drop(columns=[" Label"])
        return X, y