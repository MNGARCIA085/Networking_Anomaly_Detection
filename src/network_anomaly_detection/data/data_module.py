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

        import numpy as np

        # Quick cleaning / subset for testing
        df = (
            df.replace([np.inf, -np.inf], np.nan)
              .dropna()
              .head(500)
        )

        df.columns = df.columns.str.strip()

        X = df.drop(columns=["Label"])

        labels = df["Label"].astype(str).str.strip().str.upper()

        # Binary anomaly target:
        # BENIGN = normal (0), everything else = anomaly (1)
        y = (labels != "BENIGN").astype(int)

        return X, y






    @staticmethod
    def _split_features_labelsv0(df: pd.DataFrame):

        import numpy as np
        from sklearn.preprocessing import LabelEncoder

        # Quick cleaning / subset for testing
        df = (
            df.replace([np.inf, -np.inf], np.nan)
              .dropna()
              .head(500)
        )

        df.columns = df.columns.str.strip()

        X = df.drop(columns=["Label"])
        y = df["Label"]

        # Quick label encoding
        y = LabelEncoder().fit_transform(y)

        return X, y


    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Label"])

    print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
    """





    """
    def _split_features_labels(df: pd.DataFrame):


        # slicing


        #print(df.keys())
        # quick cleabnign for niw
        import numpy as np
        df = df.replace([np.inf, -np.inf], np.nan).dropna().head(3000) # take only 1000

        # 

        y = df["Label"] # corect later!!!!!!
        #X = df.drop(columns=[" Label"])
        X = df.drop(columns=["Label"])



        return X, y
    """