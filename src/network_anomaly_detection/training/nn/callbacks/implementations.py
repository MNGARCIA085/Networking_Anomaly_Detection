
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






