

from network_anomaly_detection.models.ml.isoforest.trainer import IsoForestTrainer
from network_anomaly_detection.models.nn.ae.trainer import AETrainer
#from network_anomaly_detection.models.nn.trainer import NNTrainer

TRAINER_REGISTRY = {
    "isoforest": IsoForestTrainer,
    "ae": AETrainer,
    # "vae": VAETrainer,
    #"ae": NNTrainer
}