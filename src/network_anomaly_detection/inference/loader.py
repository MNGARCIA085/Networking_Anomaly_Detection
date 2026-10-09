from pathlib import Path
import joblib

#import anomaly_detection.models.register_models; to trigger reg. if i use decorators
from network_anomaly_detection.models.registry import MODEL_REGISTRY

from .runner import InferenceRunner


from network_anomaly_detection.thresholding.thresholding import Thresholding



def _build_runner(
    model_dir,
    model_type,
):
    model_dir = Path(model_dir)

    # Preprocessing pipeline
    prep = joblib.load(
        model_dir / "preprocessing" / "prep.pkl"
    )

    # Model
    model_cls = MODEL_REGISTRY[model_type]

    model = model_cls.load(
        model_dir / "model"
    )

    # Optional thresholding
    thresholding_path = (
        model_dir / "thresholding" / "thresholding.pkl"
    )

    thresholding = (
        Thresholding.load(thresholding_path)
        if thresholding_path.exists()
        else None
    )

    return InferenceRunner(
        prep=prep,
        model=model,
        thresholding=thresholding,
    )




def load_from_config(config, model_dir):
    return _build_runner(
        model_dir=Path(model_dir),
        model_type=config["model"]["type"],
    )

