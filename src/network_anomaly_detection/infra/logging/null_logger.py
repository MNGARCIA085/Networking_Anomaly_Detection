from contextlib import nullcontext


# later -> make in inherit from an abst. class; ExpLogger


class NullLogger:

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