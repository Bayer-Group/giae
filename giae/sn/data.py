import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pathlib
from pytorch_lightning import LightningDataModule


DIR = pathlib.Path(__file__).parent.resolve()


def generate_dataset(num_digits, num_classes=10, save=True):
    idxs = np.random.randint(0, num_classes, size=(10000000, num_digits))
    counts = []
    for idxs_ in idxs:
        c = np.bincount(idxs_, minlength=num_classes)
        counts.append(c)
    counts = np.stack(counts)
    unique_distributions, unique_idxs = np.unique(counts, axis=0, return_index=True)
    orbits = idxs[unique_idxs]
    np.random.shuffle(orbits)
    if save:
        np.save(os.path.join(DIR, "{}_digits_{}_classes_orbits.npy".format(num_digits, num_classes)), orbits)
    return orbits


def generate_or_open_data(num_digits, num_classes=10, save=True):
    file_name = os.path.join(DIR, "{}_digits_{}_classes_orbits.npy".format(num_digits, num_classes))
    if os.path.isfile(file_name):
        print("loading data...")
        data = np.load(file_name)
    else:
        print("generating data...")
        data = generate_dataset(num_digits, num_classes, save)
    return data


class DigitSetDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers, num_train_samples, num_eval_samples, num_digits, num_classes, save=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.loader_fnc = DataLoader
        self.data = generate_or_open_data(num_digits, num_classes, save)
        if len(self.data) < self.num_eval_samples:
            raise ValueError("The data set has less samples ({}) than requested for evaluation ({}) alone.".format(len(self.data), self.num_eval_samples))

    def train_dataloader(self):
        data = self.data[self.num_eval_samples:self.num_eval_samples + self.num_train_samples]
        dataset = DigitSetDataset(data=data, num_samples=self.num_train_samples,
                                  num_digits=self.num_digits, num_classes=self.num_classes, train=True)
        dataloader = self.loader_fnc(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        data = self.data[:self.num_eval_samples]
        dataset = DigitSetDataset(data=data, num_samples=self.num_eval_samples,
                                  num_digits=self.num_digits, num_classes=self.num_classes, train=False)
        dataloader = self.loader_fnc(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dataloader


class DigitSetDataset(Dataset):
    def __init__(self, data, num_samples, num_digits, num_classes, train):
        self.digits = torch.diag(torch.ones(num_classes))  # e.g. one-hot encoded digits 0 to 9 if num_classes==10
        self.num_samples = num_samples
        self.num_digits = num_digits

        self.idxs = data
        if (num_samples < 100000) & train:
            self.idxs = self.idxs.repeat(int(1000000 / num_samples), axis=0)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idxs = self.idxs[item]
        x = self.digits[idxs]
        y = idxs.sum()
        #print(x.shape, x)
        return x, y
