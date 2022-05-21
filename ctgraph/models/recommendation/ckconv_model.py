import torch
from torch import nn

import numpy as np

from ctgraph import logger
from ctgraph.models.recommendation.module import RecommendationModule

from ctgraph.models.recommendation.ckconv_layer import CKConv

"""
python main.py --dataset beauty train --accelerator gpu --devices 1 --partial_save --val_epochs 1 --epochs 50 --batch_size 50 --batch_accum 1 --num_loader_workers 8 CKCONV --train_style dgsr_softmax --val_extra_n_vals 1 --loss_fn ce neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1

"""


class CKConvModel(RecommendationModule):
    @staticmethod
    def add_args(parser):
        # self.params.embedding_size
        # self.params.num_layers
        pass

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.info("[CKConv] Starting initiation.")
        self.graph = self.graph
        self.user_vocab_num = self.graph['u'].code.shape[0]
        self.item_vocab_num = self.graph['i'].code.shape[0]

        self.hidden_size = 32 # self.params.embedding_size
        self.num_layers = 0 # self.params.num_layers

        """ layers """
        # embedding
        self.user_embedding = nn.Embedding(self.user_vocab_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_vocab_num, self.hidden_size)

        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.conv_layers.append(CKConv(self.hidden_size))

        # propagate by concatting h || h(l-1)
        num_concats = 2
        self.w3 = nn.Linear(self.hidden_size*num_concats, self.hidden_size, bias=False)
        self.w4 = nn.Linear(self.hidden_size*num_concats, self.hidden_size, bias=False)

        # recommendation
        self.wP = nn.Linear(self.hidden_size, self.hidden_size*(self.num_layers+1), bias=False)

    def forward_graph(self, batch):
        # parse input
        u_code = batch['u'].code
        i_code = batch['i'].code
        edge_index = batch[('u', 'b', 'i')].edge_index

        # Step 1. Embedding
        hu = self.user_embedding(u_code) # (u, h)
        hi = self.item_embedding(i_code) # (i, h)

        # Step 2a. Prepare adjacency matrix
        edges = torch.sparse_coo_tensor(edge_index, values=torch.ones(edge_index.shape[1], device=edge_index.device), size=(len(hu), len(hi)), dtype=torch.float).coalesce()
        user_per_trans, item_per_trans = edges.indices()

        # Step 2b. Convolve over adjacency matrix
        hu_list = [hu]
        hi_list = [hi]
        for conv in self.conv_layers:

            # do convolution, get new user (hLu) en item (hLi) information
            hLu, hLi = conv(hu, hi, edges)

            hu_concat = torch.hstack((hLu, hu)).float()
            hi_concat = torch.hstack((hLi, hi)).float()
            
            # make new embedding
            hu = torch.tanh(self.w3(hu_concat))
            hi = torch.tanh(self.w4(hi_concat))

            # save user embedding at every timestep
            hu_list.append(hu)
            hi_list.append(hi)

        return hu_list, hi_list

    def predict_all_nodes(self, batch, predict_u):
        """
        Predict for all items in the graph
        """
        # propagate graph
        hu_list, hi_list = self.forward_graph(batch)

        # encode u
        predict_u_graph_embed = torch.hstack(hu_list)[predict_u]

        # encode all i
        # embedding of all nodes is just the embedding table
        predict_i_embed = self.wP(self.item_embedding.weight)

        # get the dot product of each element
        scores = predict_u_graph_embed @ predict_i_embed.T

        return scores

    def forward(self, batch, predict_u, predict_i=None, predict_i_ptr=None):
        """
        Predict for set combinations
        """
        if predict_i is None:
            return self.predict_all_nodes(batch, predict_u)

        assert predict_i is None or predict_i_ptr is not None

        # propagate graph
        hu_list, hi_list = self.forward_graph(batch)

        # recommendation

        # get u embeddings and i embeddings (2 lists that belong elementwise, contain duplicates)
        predict_u_graph_embed = torch.hstack(hu_list)[predict_u]

        # check if predict i are indices or codes
        if predict_i_ptr:
            predict_i_embed = self.wP(hi_list[0])[predict_i]
        else:
            predict_i_embed = self.wP(self.item_embedding(predict_i))

        # get the dot product of each element
        scores = torch.einsum('ij, ij->i', predict_u_graph_embed, predict_i_embed)

        # convert scores to a probability if it is likely to be bought
        predictions = torch.sigmoid(scores)

        return predictions