import time

import torch
import numpy as np

from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from sgat.models import SgatModule

from .dgsr_utils import sparse_dense_mul, pass_messages, relative_order

"""
lodewijk command
python main.py --dataset beauty train --nologger --accelerator cpu DGSR --user_max 10 --item_max 10 --embedding_size 64 --num_DGRN_layers=2 periodic --chunk_size 10000000 --skip_chunks 10

"""

class DGSR(SgatModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("Init online.")

    @staticmethod
    def add_args(parser):
        pass
    
    def forward(self, graph, predict_u, predict_i):
        print("SUCCESS!!!")
        pass



class DGSR(SgatModule):
    @staticmethod
    def add_args(parser):
        # max number of neighbours each node can have
        parser.add_argument('--user_max', type=int, default=10)
        parser.add_argument('--item_max', type=int, default=10)

        # layer settings
        parser.add_argument('--embedding_size', type=int, default=64)
        parser.add_argument('--num_DGRN_layers', type=int, default=2)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("[DGSR] Starting initiation.")

        print("GOT USER MAX", self.params.user_max)

        """ user_num, item_num, hidden_size, user_max, item_max, num_DGRN_layers """
        """ init """
        self.user_vocab_num = self.graph['u'].code.shape[0]
        self.item_vocab_num = self.graph['i'].code.shape[0]
        
        # Max number of neighbours used in sampling TODO
        self.user_max = 10
        self.item_max = 10
        
        self.hidden_size = self.params.embedding_size
        self.sqrt_d = np.sqrt(self.hidden_size)

        self.num_DGRN_layers = self.params.num_DGRN_layers
        
        """ layers """
        # embedding
        self.user_embedding = nn.Embedding(self.user_vocab_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_vocab_num, self.hidden_size)
        
        # propogation
        self.DGSRLayers = nn.ModuleList()
        for _ in range(self.num_DGRN_layers):
            self.DGSRLayers.append(DGRNLayer(self.user_vocab_num, self.item_vocab_num, self.hidden_size, self.user_max, self.item_max))
        
        # node updating
        self.w3 = nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)
        self.w4 = nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)
        
        # recommendation
        self.wP = nn.Linear(self.hidden_size, self.hidden_size*(self.num_DGRN_layers+1), bias=False)

        print("[DGSR] Succesfully initialised DGSR network")
        

    def forward(self, graph):
        
        # embedding
        hu = self.user_embedding(graph['u'].x) # (u, h)
        hi = self.item_embedding(graph['i'].x) # (i, h)
        
        # parse graph
        edges = graph['u', 'bought', 'i'].edge_index
        oui = graph['u', 'bought', 'i'].oui
        oiu = graph['u', 'bought', 'i'].oiu
        
        user_per_trans, item_per_trans = edges.indices()
        
        rui = relative_order(oui, user_per_trans, n=self.user_max)
        riu = relative_order(oiu, item_per_trans, n=self.item_max)
        
        # propogation
        
        # iterate over Dynamic Graph Sequential Recommendation Layers
        hu_list = [hu]
        hi_list = [hi]
        for DGSR in self.DGSRLayers:
            hLu, hSu, hLi, hSi = DGSR(hu, hi, edges, rui, riu)
            
            # concatenate information
            hu_concat = torch.hstack((hLu, hSu, hu)).float()
            hi_concat = torch.hstack((hLi, hSi, hi)).float()
            
            # make new embedding
            hu = torch.tanh(self.w3(hu_concat))
            hi = torch.tanh(self.w4(hi_concat))
            
            # save user embedding at every timestep
            hu_list.append(hu)
            hi_list.append(hi)
        
        # recommendation
        prediction_user_embedding = torch.hstack(hu_list)
        
        scores = prediction_user_embedding @ self.wP(hi_list[0]).T
        predictions = torch.softmax(scores, 0)
        
        return predictions





class DGRNLayer(nn.Module): # Dynamic Graph Recommendation Network
    def __init__(self,
                 user_num, item_num,
                 hidden_size,
                 user_max, item_max
                ):
        super().__init__()
        """ init """
        self.user_vocab_num = user_num
        self.item_vocab_num = item_num
        
        self.user_max = user_max
        self.item_max = item_max
        
        self.hidden_size = hidden_size
        self.sqrt_d = np.sqrt(self.hidden_size)
        
        """ layers """        
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Long Term User
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Long Term Item
        
        self.w3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Short Term User
        self.w4 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Short Term Item
        
        self.pV = nn.Embedding(self.user_max, self.hidden_size) # user positional embedding
        self.pK = nn.Embedding(self.item_max, self.hidden_size) # item positional embedding
        
    def longterm(self, u_embedded, i_embedded, edge_index, rui, riu):
        # --- long term ---
        
        user_messages = self.w2(u_embedded) # (u, h)
        item_messages = self.w1(i_embedded) # (i, h)
        
        # message similarity
        e = (user_messages) @ (item_messages).T # (u, i)
        e = sparse_dense_mul(edge_index, e) # (u, i)
        
        user_per_trans, item_per_trans = edge_index.indices()
        
        # - users to items -
            
        # compute positional embeddings
        pVui = self.pV(rui)
        
        # dot product van elke pos embedding met betreffende user
        u_at_pVui = torch.einsum('ij, ij->i', user_messages[user_per_trans], pVui)
        
        # alpha is softmax(wu @ wi.T + wu @ p)
        e_ui = torch.sparse_coo_tensor(e._indices(), e._values() + u_at_pVui, e.size())        
        alphas = torch.sparse.softmax(e_ui / self.sqrt_d, dim=1) # (u, i)
        
        
        # - items to users -
        
        # compute positional embeddings
        pKiu = self.pK(riu)
        
        # dot product van elke pos embedding met betreffende user
        u_at_pKiu = torch.einsum('ij, ij->i', item_messages[item_per_trans], pKiu)
        
        # beta is softmax(wi @ wu.T + wi @ p)
        e_trans = torch.transpose(e, 0, 1)
        e_iu = torch.sparse_coo_tensor(e_trans._indices(), e_trans._values() + u_at_pKiu, e_trans.size())        
        betas = torch.sparse.softmax(e_iu / self.sqrt_d, dim=1) # (u, i)
        
        # pass messages
        longterm_hu = pass_messages(item_messages, alphas, pKiu)
        longterm_hi = pass_messages(user_messages, betas, pVui)
        
        return longterm_hu, longterm_hi
    
    def shortterm(self, u, i, e, rui, riu):
        """ TODO """
        
        # pass messages
        shortterm_hu = torch.zeros((len(u), self.hidden_size)).float()
        shortterm_hi = torch.zeros((len(i), self.hidden_size)).float()
        
        return shortterm_hu, shortterm_hi
        
        
    def forward(self, u_emb, i_emb, edge_index, rui, riu):
        # propagate information
        # longterm
        hLu, hLi = self.longterm(u_emb, i_emb, edge_index, rui, riu)
        
        # shortterm
        hSu, hSi = self.shortterm(u_emb, i_emb, edge_index, rui, riu)
        
        return hLu, hSu, hLi, hSi