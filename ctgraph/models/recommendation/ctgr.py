import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv, GATConv, GATv2Conv, SGConv, LGConv

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
        if self.params.split_conv:
            self.convs2 = nn.Sequential()

        assert not (self.params.split_conv and self.params.homogenous)

        if self.params.convolution == 'SAGE':
            convolution = lambda: SAGEConv(self.params.embedding_size, self.params.embedding_size)
        elif self.params.convolution == 'GCN':
            assert self.params.homogenous or self.params.split_conv
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
        elif self.params.convolution == 'LG':
            assert self.params.homogenous
            convolution = lambda: LGConv()
        else:
            raise NotImplementedError()

        # current_size = input_size
        current_size = 0
        for i in range(self.params.conv_layers if self.params.convolution != 'SG' else 1):
            if not self.params.homogenous and not self.params.split_conv:
                self.convs.add_module(f"conv_{i}", HeteroConv({
                    ('u', 'b', 'i'): convolution(),
                    ('i', 'rev_b', 'u'): convolution(),
                }))
            elif self.params.homogenous:
                self.convs.add_module(f"conv_{i}", convolution())
            elif self.params.split_conv:
                self.convs.add_module(f"conv_{i}", convolution())
                self.convs2.add_module(f"conv2_{i}", convolution())

        if self.params.convolution == 'SG':
            # SGConv doesn't store intermediary convolutions so we just have the initial and the final onea
            self.transform = nn.Linear(self.params.embedding_size, self.params.embedding_size * 2)
        else:
            # for the dot product at the end between the complete customer embedding and a candidate article
            self.transform = nn.Linear(self.params.embedding_size,
                                   self.params.embedding_size * (self.params.conv_layers + 1))

        if self.params.activation == 'none':
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
        parser.add_argument('--split_conv', action='store_true',
                            help='Manually simulate heterogenity for convolution operators that do not support it')

    def forward(self, graph, predict_u, predict_i=None, predict_i_ptr=None):
        assert predict_i is None or predict_i_ptr is not None

        if not self.params.homogenous and not self.params.split_conv:
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
            layered_embeddings_u = [x_dict['u'][predict_u]]
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: self.activation(x) for key, x in x_dict.items()}

                layered_embeddings_u.append(x_dict['u'][predict_u])

            layered_embeddings_u = torch.cat(layered_embeddings_u, dim=1)

            # layered_embeddings_u = layered_embeddings_u[predict_u]

        elif self.params.homogenous:
            hetero_graph = HeteroData()

            # to_homogeneous needs it in 'x'
            hetero_graph['u'].x = self.user_embedding(graph['u'].code)
            hetero_graph['i'].x = self.item_embedding(graph['i'].code)
            hetero_graph['u', 'b', 'i'].edge_index = graph['u', 'b', 'i'].edge_index

            homo_graph = hetero_graph.to_homogeneous('x')

            # Ensure the users are first
            assert homo_graph.node_type.sum() == len(graph['i'].code)
            assert homo_graph.edge_type[0] == 0

            x = homo_graph.x
            edge_index = homo_graph.edge_index


            # TODO: edge_attr_dict with positional embeddings and such for GAT

            # TODO: Treat articles and users symmetrically: get layered embedding for both
            layered_embeddings_u = [x[predict_u]]
            for conv in self.convs:
                x = conv(x, edge_index)
                x = self.activation(x)
                layered_embeddings_u.append(x[predict_u])

            # Grab the layered embeddings from the users to predict. Since the concatenation by to_homogenous puts
            # them in the front we don't need to translate their indices
            # Note: For SGConv this is not really a layered embedding but just the output and input embeddings
            layered_embeddings_u = torch.cat(layered_embeddings_u, dim=1)

        elif self.params.split_conv:
            x_dict = {
                'u': self.user_embedding(graph['u'].code),
                'i': self.item_embedding(graph['i'].code)
            }
            edge_index = graph['u', 'b', 'i'].edge_index

            n_users = graph['u'].code.shape[0]

            layered_embeddings_u = [x_dict['u'][predict_u]]
            for conv, conv2 in zip(self.convs, self.convs2):
                x = torch.cat((x_dict['u'], x_dict['i']))
                x_dict['u'] = conv(x, edge_index)[:n_users]
                x_dict['i'] = conv2(x, edge_index)[n_users:]

                x_dict = {key: self.activation(x) for key, x in x_dict.items()}

                layered_embeddings_u.append(x_dict['u'][predict_u])

            layered_embeddings_u = torch.cat(layered_embeddings_u, dim=1)


        # check if predict i are indices or codes
        if predict_i is not None and predict_i_ptr:
            embeddings_i = self.item_embedding(graph['i'].code[predict_i])
        elif predict_i is not None:
            embeddings_i = self.item_embedding(predict_i)
        else:
            embeddings_i = self.item_embedding.weight

        if predict_i is not None:
            predictions = torch.sum(layered_embeddings_u * self.transform(embeddings_i), dim=1)
            # predictions = torch.dot(layered_embeddings_u, self.transform(embeddings_i))
            # predictions = torch.einsum('ij, ij->i', layered_embeddings_u, self.transform(embeddings_i))
        else:
            # Do every user for every item, resulting in a u x i matrix instead
            predictions = layered_embeddings_u @ self.transform(embeddings_i).T

        return predictions
