import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from pytorch_lightning import LightningDataModule



class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers, num_eval_samples, file_path):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_eval_samples = num_eval_samples
        self.loader_fnc = DataLoader
        self.data = None
        self.num_samples = None
        self.file_path = file_path

    def setup(self, stage=None):
        self.data = np.loadtxt(self.file_path)
        self.num_samples = len(self.data)

    def train_dataloader(self):
        dataset = MNISTDataset(self.data[self.num_eval_samples:])
        dataloader = self.loader_fnc(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = MNISTDataset(self.data[:self.num_eval_samples])
        dataloader = self.loader_fnc(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dataloader


class MNISTDataset(Dataset):
    def __init__(self, data):
        self.x = data[:, :-1].reshape(len(data), 28, 28)
        self.y = data[:, -1]
        self.num_samples = len(self.x)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        x = torch.from_numpy(self.x[item]).float()
        y = torch.from_numpy(np.array(self.y[item])).float()
        return x, y
