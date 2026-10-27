import numpy as np

from network_anomaly_detection.models.nn.ae.trainer import AETrainer


class AEIsolationForestTrainer:

    def __init__(self, model, cfg, checkpoint_dir=None):
        self.model = model
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir

        self.ae_trainer = None
        self.history = None

    def fit(
        self,
        X_train,
        y_train=None,
        X_val=None,
        y_val=None,
        ckpt_path=None,
    ):
        """
        Train the AE, then fit Isolation Forest on its latent representations.

        Training data must already contain only normal samples.
        Inputs must already have been adapted by the model.
        """

        # 1. Validate the training-data selection.
        if y_train is not None:
            labels = np.asarray(y_train).reshape(-1)

            if np.any(labels != 0):
                raise ValueError(
                    "AEIsolationForestTrainer expects normal-only "
                    "training data. Select normal samples before "
                    "calling fit()."
                )

        # 2. Train the AE using the existing Lightning trainer.
        self.ae_trainer = AETrainer(
            model=self.model.ae,
            cfg=self.cfg,
            checkpoint_dir=self.checkpoint_dir,
        )

        self.ae_trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ckpt_path=ckpt_path,
        )

        # Preserve the existing experiment logging interface.
        self.history = self.ae_trainer.history

        # 3. Fit Isolation Forest on the AE's latent representations.
        # fit detector first encodes X_train
        self.model.fit_detector(X_train)

        return self.model



    def get_config(self):
        return {
            "trainer_type": self.__class__.__name__,
            "ae_training": self.ae_trainer.get_config()
            if self.ae_trainer is not None
            else None,
            "detector": {
                "name": "IsolationForest",
            },
        }