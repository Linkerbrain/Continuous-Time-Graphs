import torch
import numpy as np

from torch import nn

from ctgraph.models.recommendation.module import RecommendationModule
from ctgraph.models.recommendation import continuous_embedding

from .dgsr_utils import relative_order
from .dgsr_layer import DGSRLayer

from ctgraph import logger
# import sys
# sys.tracebacklimit = 2

from ctgraph.models.recommendation.continuous_embedding import ContinuousTimeEmbedder

"""
lodewijk command
python main.py --dataset beauty train --accelerator gpu --devices 1 --partial_save --val_epochs 1 --epochs 20 --batch_size 25 --batch_accum 2 --num_loader_workers 8 DGSR --edge_attr none --embedding_size 50 --num_DGRN_layers 3 --train_style dgsr_softmax --loss_fn ce neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1
python main.py --dataset beauty train --accelerator gpu --devices 1 --partial_save --val_epochs 1 --epochs 20 --batch_size 10 --batch_accum 5 --num_loader_workers 5 DGSR --edge_attr positional --embedding_size 50 --num_DGRN_layers 3 --train_style dgsr_softmax --loss_fn ce neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1
python main.py --dataset beauty train --accelerator gpu --devices 1 --partial_save --val_epochs 1 --epochs 20 --batch_size 25 --batch_accum 2 --num_loader_workers 8 DGSR --edge_attr continuous --edge_attr continuous --num_siren_layers 1 --dim_hidden_size 50 --embedding_size 50 --num_DGRN_layers 3 --train_style dgsr_softmax --loss_fn ce neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1



"""

class DGSR(RecommendationModule):
    @staticmethod
    def add_args(parser):
        # layer settings
        parser.add_argument('--embedding_size', type=int, default=64)
        parser.add_argument('--num_DGRN_layers', type=int, default=2)

        parser.add_argument('--shortterm', action='store_true')
        parser.add_argument('--edge_attr', type=str, default='none', choices=['none', 'positional', 'continuous'])

        continuous_embedding.ContinuousTimeEmbedder.add_args(parser)
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.info("[DGSR] Starting initiation.")

        """ user_num, item_num, hidden_size, user_max, item_max, num_DGRN_layers """
        """ init """
        self.graph = self.graph
        self.user_vocab_num = self.graph['u'].code.shape[0]
        self.item_vocab_num = self.graph['i'].code.shape[0]

        if self.params.n_max_trans is None:
            raise ValueError("n (max_transactions) should be set in the dataset settings so the DGSR model can use it too")

        self.user_max = self.params.n_max_trans
        self.item_max = self.params.n_max_trans

        self.hidden_size = self.params.embedding_size
        self.sqrt_d = np.sqrt(self.hidden_size)

        self.num_DGRN_layers = self.params.num_DGRN_layers

        """ layers """
        # vocab embedding
        self.user_embedding = nn.Embedding(self.user_vocab_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_vocab_num, self.hidden_size)

        self.edge_attr = self.params.edge_attr
        # edge embedding
        if self.edge_attr == 'positional':
            self.pV = nn.Embedding(self.user_max, self.hidden_size) # user positional embedding
            self.pK = nn.Embedding(self.item_max, self.hidden_size) # item positional embedding
        elif self.edge_attr == 'continuous':
            self.ctUser = ContinuousTimeEmbedder(for_users=True, params=self.params)
            self.ctItem = ContinuousTimeEmbedder(for_users=False, params=self.params)

        # node updating
        self.shortterm = self.params.shortterm

        if self.shortterm:
            logger.info("[DGSR] Using shortterm too")
        else:
            logger.info("[DGSR] using no Shortterm messages")


        # propogation
        self.DGSRLayers = nn.ModuleList()
        for _ in range(self.num_DGRN_layers):
            self.DGSRLayers.append(DGSRLayer(self.user_vocab_num, self.item_vocab_num, self.hidden_size, self.user_max, self.item_max, self.shortterm, self.edge_attr, self.params))


        num_concats = 3 if self.shortterm else 2
        self.w3 = nn.Linear(self.hidden_size*num_concats, self.hidden_size, bias=False)
        self.w4 = nn.Linear(self.hidden_size*num_concats, self.hidden_size, bias=False)

        # recommendation
        self.wP = nn.Linear(self.hidden_size, self.hidden_size*(self.num_DGRN_layers+1), bias=False)

    def forward_graph(self, batch):
        u_code = batch['u'].code
        i_code = batch['i'].code
        edge_index = batch[('u', 'b', 'i')].edge_index

        oui = batch[('u', 'b', 'i')].oui
        oiu = batch[('u', 'b', 'i')].oiu

        last_user_user_code = batch[('last_u')].u_code
        last_user_item_code = batch[('last_u')].i_code_index
        last_item_user_code = batch[('last_i')].u_code_index
        last_item_item_code = batch[('last_i')].i_code

        last_user = (last_user_user_code, last_user_item_code)
        last_item = (last_item_item_code, last_item_user_code)

        # embedding
        hu = self.user_embedding(u_code) # (u, h)
        hi = self.item_embedding(i_code) # (i, h)

        # parse graph
        edges = torch.sparse_coo_tensor(edge_index, values=torch.ones(edge_index.shape[1], device=edge_index.device), size=(len(hu), len(hi)), dtype=torch.float).coalesce()
        user_per_trans, item_per_trans = edges.indices()

        # make edge embeddings
        if self.edge_attr == 'positional':
            rui = relative_order(oui, user_per_trans, n=self.user_max)
            riu = relative_order(oiu, item_per_trans, n=self.item_max)

            if self.randomize_time: # if we randomize the oui the conversion to relative messes up
                rui = oui
                riu = oiu

            pVui = self.pV(rui)
            pKiu = self.pK(riu)

        elif self.edge_attr == 'continuous':
            pVui = self.ctUser(batch)
            pKiu = self.ctItem(batch)

        else:
            pVui = None
            pKiu = None

        # propagation

        # iterate over Dynamic Graph Sequential Recommendation Layers
        hu_list = [hu]
        hi_list = [hi]
        for DGSR in self.DGSRLayers:
            hLu, hSu, hLi, hSi = DGSR(hu, hi, edges, pVui, pKiu, self.graph, last_user, last_item)

            # concatenate information
            if self.shortterm:
                hu_concat = torch.hstack((hLu, hSu, hu)).float()
                hi_concat = torch.hstack((hLi, hSi, hi)).float()
            else:
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
