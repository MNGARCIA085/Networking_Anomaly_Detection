

from network_anomaly_detection.models.nn.ae.prep import AEPrep
from network_anomaly_detection.models.ml.isoforest.prep import IsoPrep



PREP_REGISTRY = {
    "ae": AEPrep,
    "isoforest": IsoPrep,
}





"""
MODEL_REGISTRY = {
    "ae": {
        "model": AE,
        "prep": AEPrep,
    },
    "isoforest": {
        "model": IsoForest,
        "prep": IsoPrep,
    },
}
-> NO ACOPLAR si quiero mas flexibilidad!!

PREP_REGISTRY = {
    "ae_default": AEDefaultPrep,
    "ae_power": AEPowerPrep,
    "ae_raw": AERawPrep,
}

y elegir:

model:
  name: ae

prep:
  name: ae_power

trainer:
  name: nn


"""