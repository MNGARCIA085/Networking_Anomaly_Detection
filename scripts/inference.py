import numpy as np

from pathlib import Path

from network_anomaly_detection.inference.loader import load_from_config

import json


def main():


    #with open("mock_model_store/model_001/model/config.pkl", "r") as f:
    #    config = json.load(f)


    config = {
      "model": {
        "type": "ae",
        "version": "v1",
        "window_size": 10
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
    from network_anomaly_detection.data.data_module import DataModule
    data = DataModule(
        "data/data.csv",
        "data/data.csv",
    )

    X_train, y_train, X_val, y_val = data.load()

    feature_columns = X_train.columns



    import pandas as pd 
    X = pd.DataFrame(
        np.random.randn(20, 78),
        columns=feature_columns,
    )

    predictions = runner.predict(X)
    print(predictions)


if __name__ == "__main__":
    main()



