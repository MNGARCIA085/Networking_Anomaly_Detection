

from network_anomaly_detection.models.ml.isoforest.trainer import IsoForestTrainer
from network_anomaly_detection.models.nn.ae.trainer import AETrainer
#from network_anomaly_detection.models.nn.trainer import NNTrainer

from network_anomaly_detection.models.composite.ae_isoforest.trainer import (
    AEIsolationForestTrainer,
)

TRAINER_REGISTRY = {
    "isoforest": IsoForestTrainer,
    "ae": AETrainer,
    "ae_isoforest": AEIsolationForestTrainer,
    # "vae": VAETrainer,
    #"ae": NNTrainer
}