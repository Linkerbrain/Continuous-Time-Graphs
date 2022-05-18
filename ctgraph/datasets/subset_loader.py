import torch

import numpy as np
import pandas as pd
import numpy_indexed as npi

from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from ctgraph.graphs import add_random_eval_edges

from ctgraph.graphs import numpy_to_torch, add_oui_and_oiu, add_last

from ctgraph import logger

class SubsetLoader():
    """
    Subset Loader is a base module which allows the indexes of a subset to be converted to a Data object
    """
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self, graph, params):
        self.graph = graph
        self.transactions = graph[('u', 'b', 'i')].edge_index

        trans_t = graph[('u', 'b', 'i')].t
        trans_order = np.argsort(trans_t)

        self.ordered_trans = self.transactions[:, trans_order]
        self.ordered_trans_t = trans_t[trans_order]

        # train transactions are all transactions until test
        # self.train_transactions = ordered_trans[:, :-self.n_trans_test]
        # self.test_transactions = ordered_trans[:, -self.n_trans_test:]

        # print('train trans:', self.train_transactions.shape)
        # print('test trans:', self.test_transactions.shape)

    # noinspection PyTypeChecker
    def create_subgraph(self, x_idx, y_idx):
        """
        Makes pytorch graph data
        with the x_idx index transactions and the y_idx index transactions
        """
        # sample transactions ('u', 'b', 'i')
        subgraph = self._make_subset_graph(x_idx)

        # add target transactions 'target'
        self._add_target_from_vocab(subgraph, y_idx)

        # add eval transactions ['eval'].u_index, ['eval'].i_code
        add_random_eval_edges(graph=subgraph, num_items=self.graph['i'].code.shape[0],
                              true_u_index=subgraph['target'].u_index,
                              true_i_code=subgraph['target'].i_code)

        # add oui and oiu to ('u', 'b', 'i')
        add_oui_and_oiu(subgraph)

        #add last items and users to ('u', 'b', 'i')
        add_last(subgraph)

        # convert arrays to torch tensors
        subgraph_data = numpy_to_torch(subgraph)

        return subgraph_data

    # noinspection PyTypeChecker
    def _make_subset_graph(self, idx):
        """
        Makes graph only covering the transactions with index `idx`
        """
        subset_edges = self.ordered_trans[:, idx]
        subset_edges_t = self.ordered_trans_t[idx]

        # Index = new, Value = code
        customers = np.unique(subset_edges[0])
        articles = np.unique(subset_edges[1])

        subset_edges, subset_edges_t = self._remap_edges(customers, articles, subset_edges, subset_edges_t, remap_items=True)

        # build object
        subdata = HeteroData()

        subdata['u'].code = customers
        subdata['u'].num_nodes = len(customers)
        subdata['i'].code = articles
        subdata['i'].num_nodes = len(articles)

        # save edges and time codes
        subdata[('u', 'b', 'i')].edge_index = subset_edges
        subdata[('u', 'b', 'i')].t = subset_edges_t

        return subdata

    def _add_target_from_vocab(self, x_graph, idx, num_fake_edges=100):
        """
        Makes target with real edges idx and random fake edges
         defined as codes of vocab
        """
        subset_edges = self.ordered_trans[:, idx]
        subset_edges_t = self.ordered_trans_t[idx]

        # dont remap items since they are defined as codes
        subset_edges, subset_edges_t = self._remap_edges(x_graph['u'].code, None, subset_edges, subset_edges_t, remap_items=False)

        # Save target
        x_graph['target'].u_index = subset_edges[0]
        x_graph['target'].i_code = subset_edges[1]

        x_graph['target'].t = subset_edges_t

    def _remap_edges(self, customers, articles, edges, edges_t, remap_items=True):
        """
        Help function to map the codes to the index in the graph
        """
        # Make the edges point to the smaller set of users/items
        e0_ma = npi.indices(customers, edges[0], missing='mask')

        if remap_items:
            e1_ma = npi.indices(articles, edges[1], missing='mask')
            all_present = ~e0_ma.mask & ~e1_ma.mask
        else:
            all_present = ~e0_ma.mask

        edge_index = np.zeros((2, np.sum(all_present)), dtype=np.int64)

        edge_index[0] = e0_ma.data[all_present]

        if remap_items:
            edge_index[1] = e1_ma.data[all_present]
        else:
            edge_index[1] = edges[1, all_present]

        edges_t = edges_t[all_present]

        if np.sum(all_present) == 0:
            raise ValueError()

        return edge_index, edges_t

    # def _add_target_from_graph(self, x_graph, idx):
    #     """
    #     Makes target with real edges idx and random fake edges
    #      defined as indexes in x_graph
    #     """
    #     subset_edges = self.ordered_trans[:, idx]
    #     subset_edges = self._remap_edges(x_graph['u'].code, x_graph['i'].code, subset_edges)

    #     # make fake edges
    #     fake_edges = np.zeros_like(subset_edges)

    #     # same users as real edges
    #     fake_edges[0, :] = subset_edges[0, :]

    #     # Pick random items from all the items available in the graph.
    #     # TODO: Pick only from the local neighbourhood of each user
    #     fake_edges[1, :] = np.random.randint(x_graph['i'].code.shape[0], size=subset_edges.shape[1])

    #     print("target fake edges:", fake_edges)

    #     # Remove fake edges that are the same as real edges to prevent contradicting supervisions signals
    #     fake_edges = fake_edges[:, ~npi.contains(subset_edges, fake_edges, axis=1)]

    #     # Save target
    #     target = x_graph[('u', 's', 'i')]
    #     target.edge_index = np.concatenate([subset_edges, fake_edges], axis=1)

    #     target.label = np.ones(target.edge_index.shape[1], dtype=np.float)
    #     target.label[-fake_edges.shape[1]:] = 0
