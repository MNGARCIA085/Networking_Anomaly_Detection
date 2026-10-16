import json
import pandas as pd
import numpy as np
from pathlib import Path

from network_anomaly_detection.data.data_module import DataModule
from network_anomaly_detection.inference.loader import load_from_config



def main():


    #with open("mock_model_store/model_001/model/config.pkl", "r") as f:
    #    config = json.load(f)


    config = {
      "model": {
        "type": "ae",
        "version": "v1",
        "window_size": 1
      }
    }  


    model_dir = Path("mock_model_store/model_001")

    runner = load_from_config(
        config=config,
        model_dir=model_dir,
    )


    X = np.random.randn(20, 78)

    predictions = runner.predict(X)

    print("Input shape:", X.shape)
    print("Predictions:")
    print(predictions)


    #------------
    data_module = DataModule(
                train_path = "data/processed/v1/train.csv",
                val_path = "data/processed/v1/val.csv",
            )

    X_train, X_val, X_val, y_val = data_module.load_train_val()

    feature_columns = X_train.columns

    X = pd.DataFrame(
        np.random.randn(20, 78),
        columns=feature_columns,
    )

    predictions = runner.predict(X)
    print(predictions)


    # more realistic preds
    scores = runner.score(X_val)
    pred = runner.predict(X_val)
    print(pred)


    print("Threshold:", runner.thresholding.get_threshold())
    print("Validation score range:", scores.min(), scores.max())
    print("Predictions:", np.unique(runner.predict(X_val), return_counts=True))


if __name__ == "__main__":
    main()



