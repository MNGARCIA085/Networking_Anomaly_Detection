from contextlib import nullcontext
from .logger import ExperimentLogger


class NullLogger(ExperimentLogger):

    def start_run(self, run_name=None, tags=None):
        return nullcontext()

    def log_params(self, params):
        pass

    def log_metrics(self, metrics):
        pass

    def log_artifact(self, path, artifact_path=None):
        pass

    def log_tags(self, tags):
        pass

    def artifact_path(self, filename):
        return filename


    def log_training_history(self, history):
        pass


    def log_run(self, *args, **kwargs):
        pass