import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv, GATConv, GATv2Conv, SGConv, LGConv

from ctgraph import graphs
from ctgraph.models.recommendation import continuous_embedding
from ctgraph.models.recommendation.ckconv_kernel2 import KernelNet
from ctgraph.models.recommendation.continuous_embedding import ContinuousTimeEmbedder
from ctgraph.models.recommendation.dgsr_utils import relative_order
from ctgraph.models.recommendation.module import RecommendationModule

from .ckconv_kernel2 import KernelNet
from .gatv2_ckg import GATv2CKGConv
from .sage_ckg import SAGECKGConv

"""
python main.py --info train_attention_faithfull --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution GATv2 --edge_attr none --pwit --pit_target --num_siren_layers 1 --dim_hidden_size 50 --conv_layers 3 --activation tanh --concat_previous neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all

"""


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

        # -- Convolution --
        if self.params.convolution == 'SAGE':
            # I believe root_weight is equivalent to add self loops in other convs?
            convolution = lambda: SAGEConv(self.params.embedding_size, self.params.embedding_size,
                                           root_weight=self.params.add_self_loops)
        elif self.params.convolution == 'GCN':
            assert self.params.homogenous or self.params.split_conv
            convolution = lambda: GCNConv(self.params.embedding_size, self.params.embedding_size,
                                          add_self_loops=self.params.add_self_loops)
        elif self.params.convolution == 'GAT':
            edge_dim = self.params.embedding_size if self.params.edge_attr != 'none' else None
            convolution = lambda: GATConv(self.params.embedding_size, self.params.embedding_size, edge_dim=edge_dim,
                                          fill_value=0, heads=self.params.heads,
                                          add_self_loops=self.params.add_self_loops)
        elif self.params.convolution == 'GATv2':
            edge_dim = self.params.embedding_size if self.params.edge_attr != 'none' else None
            convolution = lambda: GATv2Conv(self.params.embedding_size, self.params.embedding_size, edge_dim=edge_dim,
                                            fill_value=0, heads=self.params.heads,
                                            add_self_loops=self.params.add_self_loops)
        elif self.params.convolution == 'SG':
            assert self.params.homogenous
            # SGConv does all the convolutions at once (parameter K)
            convolution = lambda: SGConv(self.params.embedding_size, self.params.embedding_size,
                                         K=self.params.conv_layers, add_self_loops=self.params.add_self_loops)
        elif self.params.convolution == 'LG':
            assert self.params.homogenous
            assert not self.params.add_self_loops
            convolution = lambda: LGConv()
        elif self.params.convolution == 'CKGSAGE':
            assert self.params.ckg
            convolution = lambda: SAGECKGConv(self.params.embedding_size, self.params.embedding_size,
                                      root_weight=self.params.add_self_loops, ckg=True)
        elif self.params.convolution == 'CKGGATv2':
            assert self.params.ckg
            convolution = lambda: GATv2CKGConv(self.params.embedding_size, self.params.embedding_size,
                                            fill_value=0, heads=self.params.heads, edge_dim=1,
                                            add_self_loops=self.params.add_self_loops)
        else:
            raise NotImplementedError()

        # -- stack layers --
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

        # -- fancy prediction --
        if self.params.pit:
            layered_embeddings_length = self.params.embedding_size * (self.params.conv_layers + 1)

            self.predict_W_u = nn.Linear(layered_embeddings_length, self.params.embedding_size, bias=False)
            self.predict_W_i = nn.Linear(self.params.embedding_size, self.params.embedding_size, bias=False)
            self.predict_W_tu = nn.Linear(self.params.embedding_size, self.params.embedding_size, bias=False)
            self.predict_W_ti = nn.Linear(self.params.embedding_size, self.params.embedding_size, bias=False)
            self.predict_a = nn.Linear(self.params.embedding_size, 1, bias=False)

        # -- fancy prwediction --
        if self.params.pwit:
            in_size = 1
            out_size = self.params.embedding_size * self.params.embedding_size * (self.params.conv_layers + 1)

            kernel_hidden_size = 50
            omega_0 = 30
            bias = True

            dropout = 0.3

            # to cuda is hotfix
            self.transform_kernel_creator = KernelNet(in_size, out_size, kernel_hidden_size, 'Sine', 'LayerNorm', 1,
                                                      bias, omega_0, dropout).to('cuda')

        # -- normal prediction --
        elif self.params.layered_embedding == 'cat':
            if self.params.convolution == 'SG':
                # SGConv doesn't store intermediary convolutions so we just have the initial and the final onea
                self.transform = nn.Linear(self.params.embedding_size, self.params.embedding_size * 2)
            else:
                # for the dot product at the end between the complete customer embedding and a candidate article
                self.transform = nn.Linear(self.params.embedding_size,
                                           self.params.embedding_size * (self.params.conv_layers + 1))
        elif self.params.layered_embedding == 'mean':
            self.transform = lambda x: x
        else:
            raise NotImplementedError()

        # -- embeddings --
        if self.params.edge_attr == 'positional':
            self.positional_user_embedding = nn.Embedding(self.params.n_max_trans,
                                                          self.params.embedding_size)  # user positional embedding
            self.positional_item_embedding = nn.Embedding(self.params.n_max_trans,
                                                          self.params.embedding_size)  # item positional embedding
        elif self.params.edge_attr == 'continuous':
            self.continuous_user_embedding = ContinuousTimeEmbedder(True, self.params, mode=self.params.siren_mode)
            self.continuous_item_embedding = ContinuousTimeEmbedder(False, self.params, mode=self.params.siren_mode)

        if self.params.activation == 'none':
            self.activation = lambda x: x
        else:
            self.activation = eval(f"torch.{self.params.activation}")

        if self.params.concat_previous:
            self.combine_transform = nn.Linear(self.params.embedding_size * 2, self.params.embedding_size)

        # -- dropout --
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
        parser.add_argument('--edge_attr', type=str, default='none', choices=['none', 'positional', 'continuous'])
        parser.add_argument('--layered_embedding', type=str, default='cat', choices=['cat', 'mean'])
        parser.add_argument('--add_self_loops', action='store_true', help='For the attention convolutions')
        parser.add_argument('--concat_previous', action='store_true')
        parser.add_argument('--siren_mode', type=str, default='t_max', choices=['t_min', 't_max', 'absolute'])
        parser.add_argument('--pit', action='store_true')
        parser.add_argument('--pwit', action='store_true')
        parser.add_argument('--pit_target', action='store_true')
        parser.add_argument('--ckg', action='store_true')
        continuous_embedding.ContinuousTimeEmbedder.add_args(parser)

    def forward(self, graph, predict_u, predict_i=None, predict_i_ptr=None):
        assert predict_i is None or predict_i_ptr is not None

        if not self.params.homogenous and not self.params.split_conv:
            layered_embeddings_u = self.get_layered_embeddings(graph, predict_u)
        elif self.params.homogenous:
            layered_embeddings_u = self.get_layered_embeddings_homo(graph, predict_u)
        elif self.params.split_conv:
            layered_embeddings_u = self.get_layered_embeddings_split(graph, predict_u)
        else:
            raise NotImplementedError()

        # check if predict i are indices or codes
        if predict_i is not None and predict_i_ptr:
            embeddings_i = self.item_embedding(graph['i'].code[predict_i]).to(predict_u.device)
        elif predict_i is not None:
            embeddings_i = self.item_embedding(predict_i).to(predict_u.device)
        else:
            embeddings_i = self.item_embedding.weight.to(predict_u.device)

        # time dependent prediction
        if self.params.pit:

            # Get t
            # TODO: Reorganice for performace: Call siren(t) first then index with predict_u
            if self.params.pit_target:
                if predict_i is not None:
                    assert torch.all(torch.sort(predict_u).values == predict_u)
                    # TODO: 101 is a hardcode of the number of fake items (100) + 1 (the real one)
                    # TODO: Prefferably dont hardcode here, the target t should really be added to the eval attribute
                    # TODO: during precomputation.
                    t = torch.repeat_interleave(graph['target'].t, 101)
                    assert t.shape == graph['eval'].u_index.shape
                else:
                    t = graph['target'].t

            t = graph['u'].t_max[predict_u] if not self.params.pit_target else t

            # embed ts
            pt_u = self.continuous_user_embedding.net(t.reshape((-1, 1)))
            pt_i = self.continuous_item_embedding.net(t.reshape((-1, 1)))

            # predict (if statement could be removed later)
            if predict_i is not None:
                # h_u.T W_1 e_i
                item_score = torch.sum(layered_embeddings_u * self.transform(embeddings_i), dim=1)
                # pt_u W_2 e_i
                tu_score = torch.sum(pt_u * self.predict_W_tu(embeddings_i), dim=1)
                # pt_i W_3 e_i
                ti_score = torch.sum(pt_i * self.predict_W_ti(embeddings_i), dim=1)

                # s_ui = h_u.T W_1 e_i + pt_u W_2 e_i + pt_i W_3 e_i
                predictions = item_score + tu_score + ti_score

            else:
                # h_u.T W_1 e_i
                item_scores = layered_embeddings_u @ self.transform(embeddings_i).T

                # pt_u W_2 e_i
                tu_score = pt_u @ self.predict_W_tu(embeddings_i).T

                # pt_i W_3 e_i
                ti_score = pt_i @ self.predict_W_ti(embeddings_i).T

                # s_ui = h_u.T W_1 e_i + pt_u W_2 e_i + pt_i W_3 e_i
                predictions = item_scores + tu_score + ti_score



        # time dependent weights prediction `s = u W(t) e_i`
        elif self.params.pwit:
            # Get t
            t = graph['target'].t * 100

            # Make kernel
            kernel_shape = (layered_embeddings_u.shape[-1], embeddings_i.shape[-1], t.shape[0])
            transform = self.transform_kernel_creator(t.unsqueeze(-1)).view(kernel_shape)

            if predict_i is not None:
                # Every user with their respective item
                num_users = transform.shape[-1]
                num_items_per_user = embeddings_i.shape[0] // num_users
                kernels = torch.arange(num_users).unsqueeze(-1).expand((-1, num_items_per_user)).flatten()

                predictions = torch.einsum('ab,bca,ac->a', layered_embeddings_u, transform[:, :, kernels], embeddings_i)
            else:
                # Do every user for every item, resulting in a u x i matrix
                # (32, 200) (200, 50, 32) (500, 50) -> (32, 50)
                predictions = torch.einsum('ab,bca,cd->ad', layered_embeddings_u, transform, embeddings_i.T)



        elif predict_i is not None:
            predictions = torch.sum(layered_embeddings_u * self.transform(embeddings_i), dim=1)
            # predictions = torch.dot(layered_embeddings_u, self.transform(embeddings_i))
            # predictions = torch.einsum('ij, ij->i', layered_embeddings_u, self.transform(embeddings_i))
        else:
            # Do every user for every item, resulting in a u x i matrix instead
            predictions = layered_embeddings_u @ self.transform(embeddings_i).T

        return predictions

    def combine_layer_embeddings(self, layer_embeddings: list):
        if self.params.layered_embedding == 'cat':
            return torch.cat(layer_embeddings, dim=1)
        elif self.params.layered_embedding == 'mean':
            return sum(layer_embeddings) / len(layer_embeddings)
        else:
            raise NotImplementedError()

    def get_layered_embeddings_split(self, graph, predict_u):
        x_dict = {
            'u': self.user_embedding(graph['u'].code),
            'i': self.item_embedding(graph['i'].code)
        }
        edge_index = graph['u', 'b', 'i'].edge_index
        n_users = graph['u'].code.shape[0]
        layer_embeddings_u = [x_dict['u'][predict_u]]
        for i, conv, conv2 in zip(range(len(self.convs)), self.convs, self.convs2):
            x = torch.cat((x_dict['u'], x_dict['i']))
            x_dict_new = dict()
            x_dict_new['u'] = conv(x, edge_index)[:n_users]
            x_dict_new['i'] = conv2(x, edge_index)[n_users:]
            if self.params.concat_previous:
                x_dict = {key: self.combine_transform(torch.cat((x, x_dict[key]), dim=1)) for key, x in
                          x_dict_new.items()}
            else:
                x_dict = x_dict_new
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}
            if i != len(self.convs) - 1:
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

            layer_embeddings_u.append(x_dict['u'][predict_u])
        layered_embeddings_u = self.combine_layer_embeddings(layer_embeddings_u)
        return layered_embeddings_u

    def get_layered_embeddings_homo(self, graph, predict_u):
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
        layer_embeddings_u = [x[predict_u]]
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            if self.params.concat_previous:
                x = self.combine_transform(torch.cat((x_new, x), dim=1))
            else:
                x = x_new
            x = self.activation(x)
            # Since the concatenation by to_homogenous puts
            # users in the front we don't need to translate their indices
            layer_embeddings_u.append(x[predict_u])
            if i != len(self.convs) - 1:
                x = self.dropout(x)
        # Grab the layered embeddings from the users to predict.
        # Note: For SGConv this is not really a layered embedding but just the output and input embeddings
        layered_embeddings_u = self.combine_layer_embeddings(layer_embeddings_u)
        return layered_embeddings_u

    def get_layered_embeddings(self, graph, predict_u):
        # TODO: Add node features
        x_dict = {
            'u': self.user_embedding(graph['u'].code),
            'i': self.item_embedding(graph['i'].code)
        }
        edge_index_dict = {
            ('u', 'b', 'i'): graph['u', 'b', 'i'].edge_index,
            ('i', 'rev_b', 'u'): graph['u', 'b', 'i'].edge_index.flip(dims=(0,))
        }

        if self.params.edge_attr == 'positional':
            rui = relative_order(graph['u', 'b', 'i'].oui, graph['u', 'b', 'i'].edge_index[0],
                                 n=self.params.n_max_trans)
            riu = relative_order(graph['u', 'b', 'i'].oiu, graph['u', 'b', 'i'].edge_index[1],
                                 n=self.params.n_max_trans)

            if self.randomize_time: # if we randomize the oui the conversion to relative messes up
                rui = graph['u', 'b', 'i'].oui
                riu = graph['u', 'b', 'i'].oiu

            # u b i points to items thus updates the item embeddings and should use riu
            # i rev_b u it the other way around
            edge_attr_dict = {
                ('u', 'b', 'i'): self.positional_item_embedding(riu),
                ('i', 'rev_b', 'u'): self.positional_user_embedding(rui)
            }
        elif self.params.edge_attr == 'continuous':
            edge_attr_dict = {
                ('u', 'b', 'i'): self.continuous_item_embedding(graph),
                ('i', 'rev_b', 'u'): self.continuous_user_embedding(graph)
            }

        if self.params.ckg:
            edges_t = graph[('u', 'b', 'i')].t
            u_t = graph['u'].t_max
            i_t = graph['i'].t_max
            edge_index = graph[('u', 'b', 'i')].edge_index
            user_per_trans, item_per_trans = edge_index[0, :], edge_index[1, :]
            edge_time_dict = {
                ('u', 'b', 'i'):  u_t[user_per_trans] - edges_t,
                ('i', 'rev_b', 'u'): i_t[item_per_trans] - edges_t
            }

        # TODO: edge_attr_dict with positional embeddings and such for GAT
        # TODO: Treat articles and users symmetrically: get layered embedding for both
        layer_embeddings_u = [x_dict['u'][predict_u]]
        for i, conv in enumerate(self.convs):
            if self.params.edge_attr == 'positional' or self.params.edge_attr == 'continuous':
                # Note that GAT and GATv2 actually also put the edge attributes through a weight matrix,
                # this is unnecessary computation since the embeddings themselves are learnable too but
                # otherwise I don't think it matters
                x_dict_new = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            elif self.params.ckg:
                x_dict_new = conv(x_dict, edge_index_dict,  edge_attr_dict=edge_time_dict)
            else:
                x_dict_new = conv(x_dict, edge_index_dict)

            if self.params.concat_previous:
                x_dict = {key: self.combine_transform(torch.cat((x, x_dict[key]), dim=1)) for key, x in
                          x_dict_new.items()}
            else:
                x_dict = x_dict_new

            x_dict = {key: self.activation(x) for key, x in x_dict.items()}
            if i != len(self.convs) - 1:
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

            layer_embeddings_u.append(x_dict['u'][predict_u])

        layered_embeddings_u = self.combine_layer_embeddings(layer_embeddings_u)
        # layered_embeddings_u = layered_embeddings_u[predict_u]
        return layered_embeddings_u
