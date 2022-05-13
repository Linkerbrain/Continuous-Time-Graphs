import torch
from torch.utils.data import Dataset
from typing import Sized, Iterable

import tqdm
import os
from os import path

from torch_geometric.loader import DataLoader

class PrecomputedDataset(Dataset):
    """
    PrecomputedDataset computes the graphs and provides dataloaders and save/load options
    """
    def __init__(self, train_yielder, val_yielder, test_yielder, batch_size, noshuffle, num_workers):
        # save yielders
        self.train_yielder = train_yielder
        self.val_yielder = val_yielder
        self.test_yielder = test_yielder

        # save settings
        self.batch_size = batch_size
        self.noshuffle = noshuffle
        self.num_workers = num_workers

        # iniate loaders
        self._init_dataloaders()

    def _init_dataloaders(self):
        # shuffle train dataset by default, except if told not to
        shuffle_train = not self.noshuffle

        self.train_data, self.train_loader = self._make_dataloader(self.train_yielder, shuffle=shuffle_train)
        self.val_data, self.val_loader = self._make_dataloader(self.val_yielder, shuffle=False)
        self.test_data, self.test_loader = self._make_dataloader(self.test_yielder, shuffle=False)

    def _make_dataloader(self, yielder, shuffle):
        data_list = []

        for data in yielder():
            # Possible data processings could happen here
            data_list.append(data)
        
        data_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

        return data_list, data_loader

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    @staticmethod
    def load_from_disk(name, data_dir="./precomputed_data/"):
        location = path.join(data_dir, name)

        return torch.load(location)

    def save_to_disk(self, data_dir="./precomputed_data/"):
        location = path.join(data_dir, 'test.torch')
        if not path.exists(data_dir):
            os.mkdir(data_dir)

        torch.save(self, location)