import numpy as np
import pandas as pd

import pickle

from sgat.subset_dataset import SubsetDataset
from sgat.recent_sampler import RecentSampler

class NeighbourDataset(SubsetDataset):
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self, graph, params, *args, **kwargs):
        """
        Neighbour dataset

        inherits from SubsetDataset, which allows the function self.create_batch
        to be used to transform indexes to graphs
        """
        # Initiate subset dataset which sorts the dataset (self.ordered_trans)
        super().__init__(graph, params, *args, **kwargs)

        print("Making neighbourDataset...")

        self.rs = RecentSampler(self.ordered_trans, self.ordered_trans_t, n=20)

        self._prepare_batch_idxs()

    """
    Neighbour sampling
    """

    def _prepare_batch_idxs(self):
        users = range(10000)
        
        self.train_data_list = []
        self.val_data_list = []
        self.test_data_list = []

        for u in users:
            sample = self.rs.neighbour_idx_of(u)

            if not sample["valid"]:
                print(f"{u} does not have enough transactions, skipping...")
                continue

            self.train_data_list.append((sample['x_train'], sample['y_train']))
            self.val_data_list.append((sample['x_val'], sample['y_val']))
            self.test_data_list.append((sample['x_test'], sample['y_test']))


    """
    Generators:
    """

    def train_data(self):

        for x_idx, y_idx in self.train_data_list:
            yield self.create_batch(x_idx, y_idx)

    def val_data(self):

        for x_idx, y_idx in self.val_data_list:
            yield self.create_batch(x_idx, y_idx)

    def test_data(self):

        for x_idx, y_idx in self.test_data_list:
            yield self.create_batch(x_idx, y_idx)

    def train_data_len(self):

        return len(self.train_data_list)
