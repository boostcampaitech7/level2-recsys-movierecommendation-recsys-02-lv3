from torch.utils.data import Dataset


class DeepFMDataset(Dataset):
    def __init__(self, args, X, y):
        self.args = args
        self.X = X
        self.y = y
        self.device = args.device

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
