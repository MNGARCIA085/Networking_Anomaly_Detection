

from network_anomaly_detection.models.nn.ae.model import AEModel
from network_anomaly_detection.models.ml.isoforest.model import IsoForestModel



MODEL_REGISTRY = {
    "ae": AEModel,
    "isoforest": IsoForestModel,
}




# later chaneg way models registred