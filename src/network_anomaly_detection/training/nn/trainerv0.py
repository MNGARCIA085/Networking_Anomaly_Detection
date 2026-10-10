from torch.utils.data import DataLoader
import torch

from .dataset import AnomalyDataset
from .schemas import TrainState, TrainingConfig, TrainingHistory


class NNTrainer:

    def __init__(self, model, cfg: TrainingConfig):
        self.model = model
        self.cfg = cfg
        self.callbacks = cfg.callbacks or []
        self.history = None
        self.optimizer = cfg.optimizer
        self.criterion = cfg.loss

    # -------- build dataloader -------- #

    def build_dataloader(self, X, y=None, shuffle=False):
        return DataLoader(
            AnomalyDataset(X, y),
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
        )

    # -------- Callbacks -------- #

    def _call_callbacks(self, hook, state):
        for cb in self.callbacks:
            getattr(cb, hook, lambda x: None)(state)

    # -------- Training step -------- #

    def training_step(
        self,
        model,
        batch,
        criterion=None,
    ):
        X, y = batch

        recon = model(X)

        return criterion(recon, X)

    # -------- Train epoch -------- #

    def train_epoch(
        self,
        model,
        loader,
        optimizer,
        criterion,
    ):


        model.train()

        total_loss = 0.0

        for batch in loader:
            X, y = batch
            #X = X.to(self.cfg.device)

            optimizer.zero_grad()

            loss = self.training_step(
                model,
                (X, y),
                criterion,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        return total_loss / len(loader.dataset)

    # -------- Validation epoch -------- #

    def validate(
        self,
        model,
        loader,
        criterion,
    ):
        model.eval()

        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                X, y = batch
                X = X.to(self.cfg.device)

                loss = self.training_step(
                    model,
                    (X, y),
                    criterion,
                )

                total_loss += loss.item() * X.size(0)

        return total_loss / len(loader.dataset)

    # -------- Fit -------- #

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        start_epoch=0, # to be resume-aware
    ):
        

        #self.model.to(self.cfg.device)

        train_loader = self.build_dataloader(
            X_train,
            y_train,
            shuffle=False, # TS
        )

        val_loader = (
            self.build_dataloader(
                X_val,
                y_val,
                shuffle=False,
            )
            if X_val is not None
            else None
        )

        history = TrainingHistory()

        state = TrainState(
            model=self.model,
            optimizer=self.optimizer,
        )

        self._call_callbacks("on_train_start", state)

        for epoch in range(
                start_epoch,
                self.cfg.epochs):
            state.epoch = epoch

            self._call_callbacks("on_epoch_start", state)

            train_loss = self.train_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.criterion,
            )

            state.train_loss = train_loss
            history.append("train_loss", train_loss)

            if val_loader is not None:
                val_loss = self.validate(
                    self.model,
                    val_loader,
                    self.criterion,
                )

                state.val_loss = val_loss
                history.append("val_loss", val_loss)

            self._call_callbacks("on_epoch_end", state)

            if state.stop_training:
                break

        self._call_callbacks("on_train_end", state)

        self.history = history

        return self.model




"""
(X, y)
  ↓
training_step()
  ↓
AE: uses X, ignores y
VAE: uses X, ignores y
DeepSVDD: uses X, ignores y
classifier: uses X + y



So the distinction is:

AE training: X → reconstruction, y ignored.
Validation: X → validation loss, with y available to callbacks/metrics if needed.
Future supervised model: can actually use y in training_step.
Unsupervised model: still gets the same generic fit(X_train, y_train, X_val, y_val) 
interface.

One small point: if you want early stopping based on val_loss, the labels aren't 
actually needed by the AE itself. But keeping y_val in the pipeline 
is useful because later you may want anomaly metrics such as PR-AUC/F1 at epoch end


"""