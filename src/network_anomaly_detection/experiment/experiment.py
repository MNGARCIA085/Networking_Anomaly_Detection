from .pipeline import ExperimentPipeline


class Experiment:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.run_id = None

        self.pipeline = ExperimentPipeline(
            cfg=cfg,
            logger=logger,
        )


    def start(self, run_name=None, nested=False, parent_run_id=None):
        run = self.logger.start_run(
            run_name=run_name,
            nested=nested,
            parent_run_id=parent_run_id,
        )

        self.run_id = run.info.run_id
        self.logger.log_tags({"status": "running"})

        return run

    def run(
        self,
        prep,
        evaluator,
        X_train,
        y_train,
        X_val,
        y_val,
        run_name=None,
        nested=False,
        parent_run_id=None,
        metadata=None,
    ):
        self.start(
            run_name=run_name,
            nested=nested,
            parent_run_id=parent_run_id,
        )

        try:
            result = self.pipeline.run(
                prep=prep,
                evaluator=evaluator,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                nested=nested,
                run_name=run_name,
                metadata=metadata,
            )

            self.finish()
            return result

        except Exception as error:
            self.fail(error)
            raise

    def finish(self):
        self.logger.log_tags({
            "status": "finished",
        })

        self.logger.end_run()

    def fail(self, error):
        self.logger.log_tags({
            "status": "failed",
            "error": str(error),
        })

        self.logger.end_run()

    def resume(
        self,
        run_id,
        evaluator,
        X_train,
        y_train,
        X_val,
        y_val,
        checkpoint=None,
        run_name=None,
    ):
        try:
            result = self.pipeline.resume(
                run_id=run_id,
                evaluator=evaluator,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                checkpoint=checkpoint,
                run_name=run_name,
            )

            self.finish() #
            return result

        except Exception as error:
            self.fail(error)
            raise