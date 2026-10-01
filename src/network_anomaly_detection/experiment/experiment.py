class ExperimentPipeline:

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def run(
        self,
        prep,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        # Prep is already supplied
        X_train, y_train = prep.transform(
            X_train,
            y_train,
        )

        X_val, y_val = prep.transform(
            X_val,
            y_val,
        )

        self._save_prep(prep)

        model_cls = MODEL_REGISTRY[
            self.cfg.model_type.name
        ]

        model = model_cls(
            self.cfg.model_type.models,
            input_shape=X_train.shape[1:],
        )

        X_train = model.adapt_input(X_train)
        X_val = model.adapt_input(X_val)

        trainer_cls = TRAINER_REGISTRY[
            self.cfg.model_type.name
        ]

        trainer = trainer_cls(
            model=model,
            cfg=self.cfg.model_type.training,
            checkpoint_dir=self.logger.checkpoint_dir(),
        )

        trainer.fit(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        return {
            "prep": prep,
            "model": model,
            "trainer": trainer,
        }