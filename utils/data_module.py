import torch
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset

class Dataset(Dataset):
    def __init__(self, data_path):
        self.data = load_data(data_path)
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx].transpose(-2, -1)
        _, y = x.max(1)
        return x, y

def load_data(data_path):
    with h5py.File(data_path, 'r') as f:
        data = f['data'][:]
    return data

class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = Dataset(self.data_path)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
