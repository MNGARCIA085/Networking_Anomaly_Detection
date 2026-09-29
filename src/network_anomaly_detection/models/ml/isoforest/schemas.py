# models/isoforest/schemas.py

from dataclasses import dataclass


@dataclass
class IsoForestConfig:
    n_estimators: int = 100
    contamination: str | float = "auto"
    max_samples: str | int | float = "auto"
    max_features: str | int | float = 1.0
    bootstrap: bool = False
    random_state: int | None = None