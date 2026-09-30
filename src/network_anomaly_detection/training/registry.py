

from network_anomaly_detection.models.ml.isoforest.trainer import IsoForestTrainer
from network_anomaly_detection.models.nn.ae.trainer import AETrainer


TRAINER_REGISTRY = {
    "isoforest": IsoForestTrainer,
    "ae": AETrainer,
    # "vae": VAETrainer,
}