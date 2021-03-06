import torch
from torch.utils.data import Dataset
from typing import Sized, Iterable

from torch_geometric.data import HeteroData
from tqdm.auto import tqdm
import os
from os import path

from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from ctgraph import logger

import sys

from ctgraph.datasets.ui_hetero_data import UIHeteroData

import pathlib

DEFAULT_DATA_DIR = "./precomputed_data/"


def _item_name(item):
    return f"user_{item}.pt"


class DiskDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.size = 0

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        location = path.join(self.root_dir, _item_name(item))
        data = torch.load(location)
        return data

    def putitem(self, graph):
        item = self.size
        location = path.join(self.root_dir, _item_name(item))

        pathlib.Path(self.root_dir).mkdir(parents=True, exist_ok=True)

        torch.save(graph, location)
        self.size += 1


class PrecomputedDataset(Dataset):
    """
    PrecomputedDataset computes the graphs and provides dataloaders and save/load options
    """

    def __init__(self, train_yielder, val_yielder, test_yielder, graph,
                 partial_save, name, data_dir=DEFAULT_DATA_DIR, neptune_logger=None):
        self.name = name
        self.data_dir = data_dir

        # save yielders
        self.train_yielder = train_yielder
        self.val_yielder = val_yielder
        self.test_yielder = test_yielder

        # save graph
        self.graph = graph


        self.partial_save = partial_save

        self.neptune_logger = neptune_logger

        # iniate loaders
        self._init_datasets()

    def _init_datasets(self):
        # shuffle train dataset by default, except if told not to

        logger.info("Creating train data..")
        self.train_data = self._make_dataset(self.train_yielder,
                                                                part='train')
        logger.info("Creating validation data..")
        self.val_data = self._make_dataset(self.val_yielder, part='val')
        logger.info("Creating test data..")
        self.test_data = self._make_dataset(self.test_yielder, part='test')

    # noinspection PyUnboundLocalVariable
    def _make_dataset(self, yielder, part):
        if not self.partial_save:
            data_list = []
        else:
            disk_dataset = DiskDataset(self.data_dir + '_' + self.name + '_' + part)

        for i, data in tqdm(enumerate(yielder())):
            if self.neptune_logger is not None:
                self.neptune_logger.experiment["loading/batch"].log(i)
            if not self.partial_save:
                data_list.append(data)
            else:
                disk_dataset.putitem(data)

        return data_list if not self.partial_save else disk_dataset


    @staticmethod
    def load_from_disk(name, data_dir=DEFAULT_DATA_DIR):
        location = path.join(data_dir, name)

        dataset = torch.load(location)

        assert dataset.name == name, "Dataset moved, cant deal with this, put it back"
        assert dataset.data_dir == data_dir, "Dataset moved, cant deal with this, put it back"
        return dataset

    def save_to_disk(self):
        self.neptune_logger = None
        
        location = path.join(self.data_dir, self.name)
        if not path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        torch.save(self, location)
