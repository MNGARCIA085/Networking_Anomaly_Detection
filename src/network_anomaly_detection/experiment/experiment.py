




class Experiment:

    def __init__(
        self,
        cfg,
        logger,
        prep,
        model_cls,
        trainer_cls,
    ):
        self.cfg = cfg
        self.logger = logger

        self.prep = prep
        self.model_cls = model_cls
        self.trainer_cls = trainer_cls

        self.run_id = None

    def run(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
    ):

        run_name = self.cfg.get(
            "run_name",
            None,
        )

        run = self.logger.start_run(
            run_name=run_name,
        )

        self.run_id = run.info.run_id

        try:

            # ---------------------------------------------
            # 1. Fit preprocessing
            # ---------------------------------------------

            self.prep.fit(
                X_train,
                y_train,
            )

            # ---------------------------------------------
            # 2. Save preprocessing
            # ---------------------------------------------

            prep_path = self.logger.artifact_path(
                "prep.joblib",
                artifact_dir=(
                    self.logger.run_artifact_dir()
                    / "preprocessing"
                ),
            )

            self.prep.save(
                prep_path,
            )

            self.logger.log_artifact(
                prep_path,
                artifact_path="preprocessing",
            )

            # ---------------------------------------------
            # 3. Transform data
            # ---------------------------------------------

            X_train = self.prep.transform(
                X_train,
            )

            X_val = self.prep.transform(
                X_val,
            )

            # ---------------------------------------------
            # 4. Build model
            # ---------------------------------------------

            model = self.model_cls(
                cfg=self.cfg.model_type.models,
                input_shape=X_train.shape[1:],
            )

            # ---------------------------------------------
            # 5. Model-specific input adaptation
            # ---------------------------------------------

            X_train = model.adapt_input(
                X_train,
            )

            X_val = model.adapt_input(
                X_val,
            )

            # ---------------------------------------------
            # 6. Build trainer
            # ---------------------------------------------

            checkpoint_dir = (
                self.logger
                .checkpoint_dir()
            )

            trainer = self.trainer_cls(
                model=model,
                cfg=self.cfg.model_type.training,
                checkpoint_dir=checkpoint_dir,
            )

            # ---------------------------------------------
            # 7. Train
            # ---------------------------------------------

            trainer.fit(
                X_train,
                y_train,
                X_val,
                y_val,
            )

            # ---------------------------------------------
            # 8. Training history
            # ---------------------------------------------

            if trainer.history is not None:

                self.logger.log_training_history(
                    trainer.history,
                )

            # ---------------------------------------------
            # 9. Finished
            # ---------------------------------------------

            self.logger.log_tags({
                "status": "finished",
            })

            return model, trainer

        except Exception as e:

            self.logger.log_tags({
                "status": "failed",
                "error": str(e),
            })

            raise

        finally:

            self.logger.end_run()