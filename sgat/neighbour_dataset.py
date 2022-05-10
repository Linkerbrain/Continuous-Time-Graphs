import numpy as np
import pandas as pd

import pickle

from sgat.subset_dataset import SubsetDataset
from sgat.recent_sampler import neighbour_idx_of

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

        # we currently make dataframe into edge index and back, this can be smoothed out obv
        df = pd.DataFrame({"u": self.ordered_trans[0, :],
                            "i": self.ordered_trans[1, :],
                            "t": self.ordered_trans_t})

        self._prepare_batch_idxs()

    """
    Neighbour sampling
    """

    def _prepare_batch_idxs(self):
        users = range(100)
        
        self.train_data = []
        self.val_data = []
        self.test_data = []

        for u in users:
            sample = neighbour_idx_of(u)

            if not sample["valid"]:
                print(f"{u} does not have enough transactions, skipping...")
                continue

            self.train_data.append((sample['x_train'], sample['y_train']))
            self.val_data.append((sample['x_val'], sample['y_val']))
            self.test_data.append((sample['x_test'], sample['y_test']))


    """
    Generators:
    """

    def train_data(self):

        for x_idx, y_idx in self.train_data:
            yield self.create_batch(x_idx, y_idx)

    def val_data(self):

        for x_idx, y_idx in self.val_data:
            yield self.create_batch(x_idx, y_idx)

    def test_data(self):

        for x_idx, y_idx in self.test_data:
            yield self.create_batch(x_idx, y_idx)

    def train_data_len(self):

        return len(self.train_data)
