import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv, GATConv, GATv2Conv, SGConv

from ctgraph import graphs
from ctgraph.models.recommendation.module import RecommendationModule


class CTGR(RecommendationModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_vocab_num = self.graph['u'].code.shape[0]
        self.item_vocab_num = self.graph['i'].code.shape[0]

        # Get rid of it as I don't need it for anything else
        del self.graph

        self.user_embedding = nn.Embedding(self.user_vocab_num, self.params.embedding_size)
        self.item_embedding = nn.Embedding(self.item_vocab_num, self.params.embedding_size)

        self.convs = nn.Sequential()

        if self.params.convolution == 'SAGE':
            convolution = lambda: SAGEConv(self.params.embedding_size, self.params.embedding_size)
        elif self.params.convolution == 'GCN':
            assert self.params.homogenous
            convolution = lambda: GCNConv(self.params.embedding_size, self.params.embedding_size)
        elif self.params.convolution == 'GAT':
            convolution = lambda: GATConv(self.params.embedding_size, self.params.embedding_size,
                                          heads=self.params.heads)
        elif self.params.convolution == 'GATv2':
            convolution = lambda: GATv2Conv(self.params.embedding_size, self.params.embedding_size,
                                            heads=self.params.heads)
        elif self.params.convolution == 'SG':
            assert self.params.homogenous
            # SGConv does all the convolutions at once (parameter K)
            convolution = lambda: SGConv(self.params.embedding_size, self.params.embedding_size,
                                         K=self.params.conv_layers)
        else:
            raise NotImplementedError()

        # current_size = input_size
        current_size = 0
        for i in range(self.params.conv_layers if self.params.convolution != 'SG' else 1):
            if not self.params.homogenous:
                self.convs.add_module(f"sage_layer_{i}", HeteroConv({
                    ('u', 'b', 'i'): convolution(),
                    ('i', 'rev_b', 'u'): convolution(),
                }))
            else:
                self.convs.add_module(f"sage_layer_{i}", HeteroConv({
                    ('a', 'b', 'a'): convolution(),
                    # ('i', 'rev_b', 'u'): convolution(),
                }))

        # for the dot product at the end between the complete customer embedding and a candidate article
        self.transform = nn.Linear(self.params.embedding_size,
                                   self.params.embedding_size * (self.params.conv_layers + 1))

        if self.params.activation is 'none':
            self.activation = lambda x: x
        else:
            self.activation = eval(f"torch.{self.params.activation}")

        self.dropout = nn.Dropout(self.params.dropout)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--embedding_size', type=int, default=50)
        parser.add_argument('--conv_layers', type=int, default=4)
        parser.add_argument('--activation', type=str, default='relu')
        parser.add_argument('--convolution', type=str, default='SAGE')
        parser.add_argument('--heads', type=int, default=1)
        parser.add_argument('--homogenous', action='store_true')
        parser.add_argument('--dropout', type=float, default=0.25)

    def predict_all_nodes(self, predict_u):
        raise NotImplementedError()

    def forward(self, graph, predict_u, predict_i=None, predict_i_ptr=None):
        assert predict_i is None or predict_i_ptr is not None


        if not self.params.homogenous:
            # TODO: Add node features
            x_dict = {
                'u': self.user_embedding(graph['u'].code),
                'i': self.item_embedding(graph['i'].code)
            }
            edge_index_dict = {
                ('u', 'b', 'i'): graph['u', 'b', 'i'].edge_index,
                ('i', 'rev_b', 'u'): graph['u', 'b', 'i'].edge_index.flip(dims=(0,))
            }
        else:
            homo_graph = graphs.to_homogenous(graph)
            x_dict = {
                'a': torch.cat((self.user_embedding(graph['u'].code), self.item_embedding(graph['i'].code)))
            }
            edge_index_dict = {
                ('a', 'b', 'a'): homo_graph['a', 'b', 'a'].edge_index,
            }


        # TODO: edge_attr_dict with positional embeddings and such for GAT

        # TODO: Treat articles and users symmetrically: get layered embedding for both
        layered_embeddings_u = [x_dict['u']]
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}

            layered_embeddings_u.append(x_dict['u'])

        # Note: For SGConv this is not really a layered embedding but just the output embedding
        layered_embeddings_u = torch.cat(layered_embeddings_u, dim=1)

        # Grab the embeddings of the users and items who we will predict for
        layered_embeddings_u = layered_embeddings_u[predict_u]

        # check if predict i are indices or codes
        if predict_i_ptr:
            embeddings_i = self.item_embedding(graph['i'].code[predict_i])
        else:
            embeddings_i = self.item_embedding(predict_i)

        # predictions = torch.dot(layered_embeddings_u, self.transform(embeddings_i))
        predictions = torch.sum(layered_embeddings_u * self.transform(embeddings_i), dim=1)
        # predictions = torch.einsum('ij, ij->i', layered_embeddings_u, self.transform(embeddings_i))

        return torch.sigmoid(predictions)
