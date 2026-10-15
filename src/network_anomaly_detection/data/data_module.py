from pathlib import Path
import pandas as pd




class DataModule:
    def __init__(
        self,
        train_path: str | Path | None = None,
        val_path: str | Path | None = None,
        test_path: str | Path | None = None,
        label_column: str = "Label",
    ):
        self.paths = {
            "train": Path(train_path) if train_path is not None else None,
            "val": Path(val_path) if val_path is not None else None,
            "test": Path(test_path) if test_path is not None else None,
        }
        self.label_column = label_column

    def load_split(self, split: str):
        if split not in self.paths:
            raise ValueError(f"Unknown split: {split!r}")

        path = self.paths[split]

        if path is None:
            raise ValueError(f"No path configured for split {split!r}")

        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)
        return self._split_features_labels(df)

    def load_train_val(self):
        X_train, y_train = self.load_split("train")
        X_val, y_val = self.load_split("val")
        return X_train, y_train, X_val, y_val

    def _split_features_labels(self, df: pd.DataFrame):
        if not df.columns.is_unique:
            raise ValueError("Dataset contains duplicate columns.")

        if self.label_column not in df.columns:
            raise ValueError(
                f"Target column {self.label_column!r} not found."
            )

        X = df.drop(columns=[self.label_column])
        y = df[self.label_column]

        if X.empty:
            raise ValueError("Dataset contains no feature columns.")

        return X, y



"""
One deliberate choice: I haven't added missing-value rejection here. Your prep pipeline 
should own the strategy for handling missing features, while the DataModule simply loads 
and separates the data.
"""



