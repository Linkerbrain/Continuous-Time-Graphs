from tkinter import N
import numpy as np
import pandas as pd
import pickle

from ctgraph import logger
from ctgraph.np_utils import cumcount

def sample_neighbourhood_(edges, fromm, sample_source, mask, hops):
    """
    edges: edge_index [[u, u, u], [i, i, i]]
    fromm: sampling from x (0 if from users, 1 if from items)
    sample_source: idx of nodes to be explored
    mask: nodes that are marked as sampled
    hops: hops left to do
    """
    
    # basecase, return the transactions that were sampled
    if hops == 0:
        return mask
        
    # get latest transaction by `fromm`, remove edges that were previously sampled (`~mask`)
    new_transaction_mask = np.isin(edges[fromm], sample_source) & ~mask
    
    # if we sample from users, we sample to items and vice versa
    to = 0 if fromm else 1
    
    # next iteration we will sample from the nodes we just sampled
    new_sources = edges[to, new_transaction_mask]
    
    # add samples to our total
    mask |= new_transaction_mask

    # sample next iteration
    return sample_neighbourhood_(edges, to, new_sources, mask, hops = hops - 1)

class PaperSampler():
    """
    Samples like in the paper
    """
    def __init__(self, ordered_trans, ordered_trans_t, n, m, sample_all):
        """
        ordered_trans is edge_index [[u, u, u, u], [i, i, i, i]]
        ordered_trans_t is corresponding time [t, t, t, t]

        n is max number of recent transactions per node sampled
        """
        self.ordered_trans = ordered_trans
        self.ordered_trans_t = ordered_trans_t

        # number of max transactions per user
        self.n = n
        # max order neighbourhood to sample
        self.m = m
        # 1 hop is from users to items or vice versa. As dgsr defines m it is 2 hops + they ignore the initial hop
        self.hops = 1 + m * 2

        # TODO make parameter (?)
        self.min_trans_count = 3
        self.min_graph_size = 20 if self.m > 0 else 5

        self.sample_all = sample_all


    def sample_neighbourhood(self, t, target_user):
        """
        t: time code it needs to be before
        target_user: user to sample from
        """
        
        # remove edges from the future
        legal_mask = self.ordered_trans_t < t
        
        # only accept the most recent n transactions of items
        legal_mask[legal_mask] = (cumcount(self.ordered_trans[1, legal_mask]) < self.n)
        
        # only accept the most recent n transactions of users
        legal_mask[legal_mask] = (cumcount(self.ordered_trans[0, legal_mask]) < self.n)
        
        legal_edges = self.ordered_trans[:, legal_mask]

        legal_mask[legal_mask] = sample_neighbourhood_(legal_edges, 0, np.array([target_user]),
                                                    mask=np.zeros(legal_edges.shape[1], dtype=bool),
                                                    hops=self.hops)

        return np.where(legal_mask)[0]

    def neighbour_idx_of(self, target_u):
        # get transasctions of user
        user_trans = np.where(self.ordered_trans[0]==target_u)[0]

        if len(user_trans) < self.min_trans_count: # todo make parameter
            return {"valid":False}

        # test (final transaction is used for testing)
        target_idx = user_trans[-1]

        test_neighbourhood = self.sample_neighbourhood(self.ordered_trans_t[target_idx], target_u)
        if len(test_neighbourhood) < self.min_graph_size:
            return {"valid":False}

        y_test_trans = np.expand_dims(target_idx, 0)
        x_test_trans = test_neighbourhood

        # validating (penultimate is used for valdiating)
        target_idx = user_trans[-2]

        val_neighbourhood = self.sample_neighbourhood(self.ordered_trans_t[target_idx], target_u)
        if len(val_neighbourhood) < self.min_graph_size:
            return {"valid":False}

        y_val_trans = np.expand_dims(target_idx, 0)
        x_val_trans = val_neighbourhood

        # training (others are used for training)
        if self.sample_all:
            target_idxs = user_trans[-3::-1]
        else:
            target_idxs = [user_trans[-3]]

        x_train_trans_list = []
        y_train_trans_list = []
        for target_idx in target_idxs:
            train_neighbourhood = self.sample_neighbourhood(self.ordered_trans_t[target_idx], target_u)

            if len(train_neighbourhood) < self.min_graph_size:
                break

            y_train_trans_list.append(np.expand_dims(target_idx, 0))
            x_train_trans_list.append(train_neighbourhood)

        if len(x_train_trans_list) == 0:
            return {"valid":False}

        # build return
        return {"x_train" : x_train_trans_list, "y_train" : y_train_trans_list,
                "x_val" : [x_val_trans], "y_val" : [y_val_trans],
                "x_test" : [x_test_trans], "y_test" : [y_test_trans],
                "valid":True}
