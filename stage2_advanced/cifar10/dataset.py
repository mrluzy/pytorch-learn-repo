from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, train=True):
        # TODO: load data
        pass

    def __len__(self):
        return 0  # replace with dataset length

    def __getitem__(self, idx):
        # TODO: return sample
        return None
