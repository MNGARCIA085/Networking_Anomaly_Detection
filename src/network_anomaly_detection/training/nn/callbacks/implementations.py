
# callbacks interface
class Callback:
    
    def on_train_start(self, state): 
        pass
    
    def on_epoch_start(self, state): 
        pass
    
    def on_epoch_end(self, state): 
        pass
    
    def on_train_end(self, state): 
        pass




class PrintLossCallback(Callback):
    def on_epoch_end(self, state):
        print(f"Epoch {state.epoch} - Train Loss: {state.train_loss:.4f} - Val Loss: {state.val_loss:.4f}")


"""
class PrintLossCallback(Callback):

    def on_epoch_end(self, state):
        msg = f"Epoch {state.epoch} - Train Loss: {state.train_loss:.4f}"

        if state.val_loss is not None:
            msg += f" - Val Loss: {state.val_loss:.4f}"

        print(msg)
"""




class EarlyStopping(Callback):
    def __init__(self, patience=5):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def on_epoch_end(self, state):
        if state.val_loss is None:
            return

        if state.val_loss < self.best:
            self.best = state.val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print('ES triggreed')
            state.stop_training = True






#------------chckpoint---------------#
from pathlib import Path

import torch





from pathlib import Path

import torch



class CheckpointCallback(Callback):

    def __init__(
        self,
        directory,
        every_n_epochs=10,
        keep_last=2,
        save_best=True,
        monitor="val_loss",
        mode="min",
    ):
        self.directory = Path(directory)
        self.every_n_epochs = every_n_epochs
        self.keep_last = keep_last
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.saved_checkpoints = []
        self.best_metric = None

    def _is_better(self, metric):

        if self.best_metric is None:
            return True

        if self.mode == "min":
            return metric < self.best_metric

        return metric > self.best_metric

    def _save(self, state, filename):

        path = self.directory / filename

        torch.save(
            {
                "epoch": state.epoch,
                "model_state_dict": state.model.state_dict(),
                "optimizer_state_dict": state.optimizer.state_dict(),
                "best_metric": self.best_metric,
            },
            path,
        )

        return path

    def on_epoch_end(self, state):

        epoch = state.epoch + 1

        # Periodic checkpoint
        if epoch % self.every_n_epochs == 0:

            path = self._save(
                state,
                f"epoch_{epoch:04d}.pt",
            )

            self.saved_checkpoints.append(path)

            while len(self.saved_checkpoints) > self.keep_last:

                old_path = self.saved_checkpoints.pop(0)

                if old_path.exists():
                    old_path.unlink()

        # Best checkpoint
        if self.save_best:

            metric = getattr(
                state,
                self.monitor,
                None,
            )

            if metric is not None and self._is_better(metric):

                self.best_metric = metric

                self._save(
                    state,
                    "best.pt",
                )

    def load_state(self, checkpoint):

        self.best_metric = checkpoint.get(
            "best_metric"
        )







class CheckpointCallbackv0(Callback):

    def __init__(
        self,
        directory,
        every_n_epochs=10,
        keep_last=2,
        save_best=True,
        monitor="val_loss",
        mode="min",
    ):
        self.directory = Path(directory)
        self.every_n_epochs = every_n_epochs
        self.keep_last = keep_last
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.saved_checkpoints = []
        self.best_metric = None

    def _is_better(self, metric):

        if self.best_metric is None:
            return True

        if self.mode == "min":
            return metric < self.best_metric

        return metric > self.best_metric

    def _save(self, state, filename):

        path = self.directory / filename

        torch.save(
            {
                "epoch": state.epoch,
                "model_state_dict": state.model.state_dict(),
                "optimizer_state_dict": state.optimizer.state_dict(),
            },
            path,
        )

        return path

    def on_epoch_end(self, state):

        epoch = state.epoch + 1

        # Periodic checkpoint
        if epoch % self.every_n_epochs == 0:

            path = self._save(
                state,
                f"epoch_{epoch:04d}.pt",
            )

            self.saved_checkpoints.append(path)

            while len(self.saved_checkpoints) > self.keep_last:

                old_path = self.saved_checkpoints.pop(0)

                if old_path.exists():
                    old_path.unlink()

        # Best checkpoint
        if self.save_best:

            metric = getattr(
                state,
                self.monitor,
                None,
            )

            if metric is not None and self._is_better(metric):

                self.best_metric = metric

                self._save(
                    state,
                    "best.pt",
                )



"""
This gives you:

checkpoints/
├── best.pt
├── epoch_0030.pt
└── epoch_0040.pt

with keep_last=2.
"""