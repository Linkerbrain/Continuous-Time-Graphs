from tkinter import N
import numpy as np
import pandas as pd
import pickle

from ctgraph import logger

class RecentSampler():
    """
    RecentSampler samples the most recent m-order neighbourhood of an user efficiently
    """
    def __init__(self, ordered_trans, ordered_trans_t, n, m):
        """
        ordered_trans is edge_index [[u, u, u, u], [i, i, i, i]]
        ordered_trans_t is corresponding time [t, t, t, t]

        n is max number of recent transactions per node sampled
        """
        self.ordered_trans = ordered_trans
        self.ordered_trans_t = ordered_trans_t

        # we currently make dataframe into edge index and back, this can be smoothed out obv
        self.df = pd.DataFrame({"u": self.ordered_trans[0, :],
                            "i": self.ordered_trans[1, :],
                            "t": self.ordered_trans_t})

        # number of max transactions per user
        self.n = n
        # max order neighbourhood to sample
        self.m = m

        self.u_connections, self.u_transactions, self.i_connections, self.i_transactions = self._make_dictionaries(self.df, n)

    def _make_dictionaries(self, df, n=20):
        # -- build User most recent transaction lists --
        u_connection_list = [np.zeros(n)]
        u_transaction_list = [np.zeros(n)]

        for u, us_transactions in df.groupby('u'):
            bought = us_transactions['i'].values[-n:]
            
            zero_padded = np.zeros(n)
            zero_padded[:len(bought)] = bought + 1 # offset by 1 for dummy
            
            u_connection_list.append(zero_padded)
            
            transaction_idx = us_transactions['i'].index.values[-n:]
            
            zero_padded_t = np.zeros(n)
            zero_padded_t[:len(transaction_idx)] = transaction_idx
            
            u_transaction_list.append(zero_padded_t)

        # -- build Item most recent transaction lists --
        i_connection_list = [np.zeros(n)]
        i_transaction_list = [np.zeros(n)]

        for i, is_transactions in df.groupby('i'):
            bought = is_transactions['u'].values[-n:]
            
            zero_padded = np.zeros(n)
            zero_padded[:len(bought)] = bought + 1 # offset by 1 for dummy
            
            i_connection_list.append(zero_padded)
            
            transaction_idx = is_transactions['u'].index.values[-n:]
            
            zero_padded_t = np.zeros(n)
            zero_padded_t[:len(transaction_idx)] = transaction_idx
            
            i_transaction_list.append(zero_padded_t)

        # -- parse to array --

        u_connections = np.stack(u_connection_list).astype(np.int32)
        u_transactions = np.stack(u_transaction_list).astype(np.int32)

        i_connections = np.stack(i_connection_list).astype(np.int32)
        i_transactions = np.stack(i_transaction_list).astype(np.int32)

        return u_connections, u_transactions, i_connections, i_transactions

    # get user network

    def get_user_network(self, index):
        # u_m and i_m are the sets of explored nodes
        u_m = np.array([0]) # 0 is dummy
        i_m = np.array([0])
        
        # transactions of sampled nodes
        transactions_m = np.array([0])
        
        # u_temp and i_temp are the sets of unexplored nodes
        u_temp = np.array([index+1]) # initialize as the given index
        i_temp = self.u_connections[u_temp] # initialize as its purchases
        
        # add initialized purchases to transaction base
        new_transactions = self.u_transactions[u_temp].flatten()
        transactions_m = np.union1d(transactions_m, new_transactions)
            
        for j in range(self.m):
            new_users = np.unique(self.i_connections[i_temp])
            u_temp = np.union1d(u_temp, new_users)
            
            new_transactions = self.i_transactions[i_temp].flatten()
            transactions_m = np.union1d(transactions_m, new_transactions)
            
            u_temp = np.setdiff1d(u_temp, u_m, assume_unique=True)
            u_m = np.union1d(u_m, u_temp)
            
            if len(u_temp)==0:
                break
                
            new_items = np.unique(self.u_connections[u_temp])
            i_temp = np.union1d(i_temp, new_items)
            
            new_transactions = self.u_transactions[u_temp].flatten()
            transactions_m = np.union1d(transactions_m, new_transactions)
            
            i_temp = np.setdiff1d(i_temp, i_m, assume_unique=True)
            i_m = np.union1d(i_temp, i_m)
            
            if len(i_temp)==0:
                break
        
        # [1:] to ignore first element since its dummy 0
        # -1 to offset back (it was offset to allow for dummy 0)
        return u_m[1:]-1, i_m[1:]-1, transactions_m[1:]

    def neighbour_idx_of(self, target_u):
        # get subset
        u, i, t_idxs = self.get_user_network(target_u)

        # choose most recent item for testing
        # second most recent item for validation
        # other items for training

        sub = self.ordered_trans[:, t_idxs]

        user_trans = np.where(sub[0]==target_u)[0]

        if len(user_trans) < 5: # todo make parameter
            return {"valid":False}

        # testing
        final_idx = user_trans[-1]

        y_test_trans = np.expand_dims(t_idxs[final_idx], 0)
        x_test_trans = t_idxs[:final_idx]

        # validating
        pen_ult_idx = user_trans[-2]
        y_val_trans = np.expand_dims(t_idxs[pen_ult_idx], 0)
        x_val_trans = t_idxs[:pen_ult_idx]

        # training (currently only -3, but we could do everything tot en met -3)
        pen_pen_ult_idx = user_trans[-3]
        y_train_trans = np.expand_dims(t_idxs[pen_pen_ult_idx], 0)
        x_train_trans = t_idxs[:pen_pen_ult_idx]

        # build return
        return {"x_train" : x_train_trans, "y_train" : y_train_trans,
                "x_val" : x_val_trans, "y_val" : y_val_trans,
                "x_test" : x_test_trans, "y_test" : y_test_trans,
                "valid":True}

if __name__ == "__main__":
    with open("./debug_ordered_trans.pickle", mode="rb") as f:
        ordered_trans = pickle.load(file=f)

    with open("./debug_ordered_trans_t.pickle", mode="rb") as f:
        ordered_trans_t = pickle.load(file=f)

    target_u = 41

    df = pd.DataFrame({'u':ordered_trans[0, :], 'i':ordered_trans[1, :], 't':ordered_trans_t})

    rs = RecentSampler(df, n=20)

    sampled = rs.neighbour_idx_of(target_u)

    print(sampled)