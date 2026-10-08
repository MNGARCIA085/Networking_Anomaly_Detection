from network_anomaly_detection.models.nn.ae.tuning import AESampler
from network_anomaly_detection.models.ml.isoforest.tuning import IsoForestSampler


TUNING_REGISTRY = {
    "ae": AESampler,
    "isoforest": IsoForestSampler,
}