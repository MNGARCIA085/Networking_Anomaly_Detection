from abc import ABC, abstractmethod


class ExperimentLogger(ABC):

    @abstractmethod
    def start_run(self, run_name=None):
        pass

    @abstractmethod
    def end_run(self):
        pass

    @abstractmethod
    def log_params(self, params):
        pass

    @abstractmethod
    def log_metrics(self, metrics):
        pass

    @abstractmethod
    def log_artifact(self, path):
        pass

    @abstractmethod
    def log_model(self, model, path):
        pass

    # add missing methods