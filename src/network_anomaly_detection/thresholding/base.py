from abc import ABC, abstractmethod

class ThresholdStrategy(ABC):

    @abstractmethod
    def fit(
        self,
        scores=None,
        y_val=None,
    ):
        pass

    @abstractmethod
    def get_threshold(self):
        pass


    @abstractmethod
    def get_config(self):
        pass



"""
Thresholding
  fit(scores)            → calculates/stores threshold
  get_threshold()        → returns threshold
"""
