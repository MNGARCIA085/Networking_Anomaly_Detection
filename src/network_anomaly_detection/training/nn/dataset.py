from torch.utils.data import Dataset
import torch

class AnomalyDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)

        self.y = (
            torch.tensor(y, dtype=torch.long)
            if y is not None
            else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]

        return self.X[idx], self.y[idx]