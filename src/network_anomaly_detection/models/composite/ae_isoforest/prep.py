

"""
for now, ill reuse aes

better code for mask selection

def select_training_data(self, X_train, y_train):
    mask = y_train == 0
    X_train = X_train.loc[mask] if hasattr(X_train, "loc") else X_train[mask]
    y_train = y_train.loc[mask] if hasattr(y_train, "loc") else y_train[mask]
    return X_train, y_train

"""