from network_anomaly_detection.models.nn.ae.tuning import AESampler
from network_anomaly_detection.models.nn.vae.tuning import VAESampler
from network_anomaly_detection.models.ml.isoforest.tuning import IsoForestSampler
from network_anomaly_detection.models.composite.ae_isoforest.tuning import AEIsolationForestSampler
from network_anomaly_detection.models.nn.transformer.tuning import TransformerAESampler


TUNING_REGISTRY = {
    "ae": AESampler,
    "vae": VAESampler,
    "isoforest": IsoForestSampler,
    "ae_isoforest": AEIsolationForestSampler,
    "transformer": TransformerAESampler,
}