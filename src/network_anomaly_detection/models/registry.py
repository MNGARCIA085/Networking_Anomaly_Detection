

from network_anomaly_detection.models.nn.ae.model import AEModel
from network_anomaly_detection.models.ml.isoforest.model import IsoForestModel
from network_anomaly_detection.models.composite.ae_isoforest.model import AEIsolationForestModel


from network_anomaly_detection.models.nn.transformer.model import TransformerAEModel


MODEL_REGISTRY = {
    "ae": AEModel,
    "isoforest": IsoForestModel,
    "ae_isoforest": AEIsolationForestModel,
    "transformer": TransformerAEModel,
}




# later chaneg way models registred