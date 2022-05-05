import torch

import numpy as np
import pandas as pd
import numpy_indexed as npi

from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

class SimpleDataLoaders(Dataset):
    def __init__(self, graph, params):
        print("Graph:", graph)

        self.transactions = graph[('u', 'b', 'i')].edge_index

        self.n_trans = self.transactions.shape[1] # 394 908 for beauty
        self.n_trans_test = 1000
        self.n_trans_train = self.n_trans - self.n_trans_test

        # rolling window x, y, v over train transactions
        # taking first x trans for embedding, then the y after for loss, then the eval for evaluation
        self.n_trans_x = 10000
        self.n_trans_y = 1000
        self.n_trans_eval = 1000

        trans_t = graph[('u', 'b', 'i')].t
        trans_order = np.argsort(trans_t)

        self.ordered_trans = self.transactions[:, trans_order]

        # train transactions are all transactions until test
        # self.train_transactions = ordered_trans[:, :-self.n_trans_test]
        # self.test_transactions = ordered_trans[:, -self.n_trans_test:]

        # print('train trans:', self.train_transactions.shape)
        # print('test trans:', self.test_transactions.shape)

    def create_batch(self, x_idx, y_idx):
        """
        Makes graph with the x_idx index transactions,
        and target with the y_idx index transactions
        """
        # sample transactions
        x_graph = self._make_subset_graph(x_idx)

        # ren
        self._add_target(x_graph, y_idx)

        return x_graph

    def _make_subset_graph(self, idx):
        """
        Makes graph only covering the transactions with index `idx`
        """
        subset_edges = self.ordered_trans[:, idx]

        # Index = new, Value = code
        customers = np.unique(subset_edges[0])
        articles = np.unique(subset_edges[1])

        subset_edges = self._remap_edges(customers, articles, subset_edges)

        # build object
        subdata = HeteroData()

        subdata['u'].code = customers
        subdata['i'].code = articles

        subdata[('u', 'b', 'i')] = subset_edges

        return subdata

    def _add_target(self, x_graph, idx):
        """
        Makes target with real edges idx and random fake edges
         defined as indexes in x_graph
        """
        # 
        subset_edges = self.ordered_trans[:, idx]

        customers = npi.indices(x_graph['u'].code, subset_edges[0])
        articles = npi.indices(x_graph['i'].code, subset_edges[1])

        subset_edges = self._remap_edges(customers, articles, subset_edges)

        # make fake edges
        fake_edges = np.zeros_like(subset_edges)

        # same users as real edges
        fake_edges[0, :] = subset_edges[0, :]

        # Pick random items from all the items available in the graph.
        # TODO: Pick only from the local neighbourhood of each user
        fake_edges[1, :] = np.random.randint(x_graph['i'].code.shape[0], size=subset_edges.shape[1])

        # Remove fake edges that are the same as real edges to prevent contradicting supervisions signals
        fake_edges = fake_edges[:, ~npi.contains(subset_edges, fake_edges, axis=1)]

        # Save target
        target = x_graph[('u', 's', 'i')]
        target.edge_index = np.concatenate([subset_edges, fake_edges], axis=1)

        target.label = np.ones(target.edge_index.shape[1], dtype=np.float)
        target.label[-fake_edges.shape[1]:] = 0


    def _remap_edges(self, customers, articles, edges):
        """
        Help function to map the codes to the index in the graph
        """
        # Make the edges point to the smaller set of users/items
        e0_ma = npi.indices(customers, edges[0], missing='mask')
        e1_ma = npi.indices(articles, edges[1], missing='mask')

        all_present = ~e0_ma.mask & ~e1_ma.mask
        
        edge_index = np.zeros((2, np.sum(all_present)), dtype=np.int64)
        edge_index[0] = e0_ma.data[all_present]
        edge_index[1] = e1_ma.data[all_present]

        return edge_index




    @staticmethod
    def add_args(parser):
        pass