from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):


    @abstractmethod
    def adapt_input(self, X: np.ndarray) -> np.ndarray:
        """Adapt canonical preprocessed data to model input."""
        pass


    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return an anomaly score for each input sample."""
        pass

    def predict(
        self,
        X: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        scores = self.score(X)
        return scores >= threshold

    

    """
    save method
    """