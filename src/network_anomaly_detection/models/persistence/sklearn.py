from pathlib import Path
import joblib

def save_sklearn_model(model, path):

    path = Path(path)
    path.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        path / "model.pkl"
    )


def load_sklearn_model(path):

    path = Path(path)

    return joblib.load(
        path / "model.pkl"
    )