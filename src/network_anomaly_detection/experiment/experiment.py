
from .pipeline import ExperimentPipeline

#
# diagram of lifecycle!!!




class Experiment:

    def __init__(
        self,
        cfg,
        logger,
    ):
        self.cfg = cfg
        self.logger = logger
        self.run_id = None

        self.pipeline = ExperimentPipeline(
            cfg=cfg,
            logger=logger,
        )

    def start(self, run_name=None):

        run = self.logger.start_run(
            run_name=run_name,
        )

        self.run_id = run.info.run_id

        self.logger.log_tags({
            "status": "running",
        })

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
    ):

        self.start(
            run_name=run_name,
        )

        try:

            result = self.pipeline.run(
                prep=prep,
                evaluator=evaluator,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
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


    # resume
    def resume(
        self,
        run_id,
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
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                checkpoint=checkpoint,
                run_name=run_name,
            )

            return result

        except Exception as error:

            self.fail(error)

            raise


   
