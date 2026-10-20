

from network_anomaly_detection.models.ml.isoforest.trainer import IsoForestTrainer
from network_anomaly_detection.models.nn.ae.trainer import AETrainer


from network_anomaly_detection.models.composite.ae_isoforest.trainer import (
    AEIsolationForestTrainer,
)


from network_anomaly_detection.models.nn.transformer.trainer import TransformerTrainer


TRAINER_REGISTRY = {
    "isoforest": IsoForestTrainer,
    "ae": AETrainer,
    "ae_isoforest": AEIsolationForestTrainer,
    "transformer": TransformerTrainer,
    # "vae": VAETrainer,
}