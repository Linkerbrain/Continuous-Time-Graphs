import time

import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from sgat.models.sgat_module import SgatModule

class MH(SgatModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_vocab_num = self.graph['u'].code.shape[0]
        self.item_vocab_num = self.graph['i'].code.shape[0]

        # Get rid of it as I don't need it for anything else
        del self.graph

        self.user_embedding = nn.Embedding(self.user_vocab_num, self.params.embedding_size)
        self.item_embedding = nn.Embedding(self.item_vocab_num, self.params.embedding_size)

        self.convs = nn.Sequential()

        # current_size = input_size
        current_size = 0
        for i in range(self.params.conv_layers):
            self.convs.add_module(f"sage_layer_{i}", HeteroConv({
                ('u', 'b', 'i'): SAGEConv(self.params.embedding_size, self.params.embedding_size),
                ('i', 'rev_b', 'u'): SAGEConv(self.params.embedding_size, self.params.embedding_size),
            }))

        # for the dot product at the end between the complete customer embedding and a candidate article
        self.transform = nn.Linear(self.params.embedding_size, self.params.embedding_size * (self.params.conv_layers+1))

        if self.params.activation is None:
            self.activation = lambda x: x
        else:
            self.activation = eval(f"torch.{self.params.activation}")

        self.dropout = nn.Dropout(self.params.dropout)



    @staticmethod
    def add_args(parser):
        parser.add_argument('--embedding_size', type=int, default=50)
        parser.add_argument('--conv_layers', type=int, default=4)
        parser.add_argument('--activation', type=str, default=None)
        parser.add_argument('--dropout', type=float, default=0.25)

    def forward(self, graph, predict_u, predict_i, predict_i_ptr=True):
        assert predict_i_ptr

        # TODO: Add node features
        x_dict = {
            'u': self.user_embedding(graph['u'].code),
            'i': self.item_embedding(graph['i'].code)
        }

        edge_index_dict = {
            ('u', 'b', 'i'): graph['u', 'b', 'i'].edge_index,
            ('i', 'rev_b', 'u'): graph['u', 'b', 'i'].edge_index.flip(dims=(0,))
        }

        # TODO: edge_attr_dict with positional embeddings and such for GAT

        # TODO: Treat articles and users symmetrically: get layered embedding for both
        layered_embeddings_u = [x_dict['u']]
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}

            layered_embeddings_u.append(x_dict['u'])
        layered_embeddings_u = torch.cat(layered_embeddings_u, dim=1)

        # Grab the embeddings of the users and items who we will predict for
        layered_embeddings_u = layered_embeddings_u[predict_u]

        # check if predict i are indices or codes
        if predict_i_ptr:
            embeddings_i = x_dict['i'][predict_i]
        else:
            embeddings_i = self.item_embedding(predict_i)

        # predictions = torch.dot(layered_embeddings_u, self.transform(embeddings_i))
        predictions = torch.sum(layered_embeddings_u * self.transform(embeddings_i), dim=1)
        # predictions = torch.einsum('ij, ij->i', layered_embeddings_u, self.transform(embeddings_i))

        return torch.sigmoid(predictions)
