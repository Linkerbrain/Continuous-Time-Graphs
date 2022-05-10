import torch
import torch.nn as nn
import numpy as np

from .dgsr_utils import sparse_dense_mul, pass_messages, relative_order

class DGSRLayer(nn.Module): # Dynamic Graph Recommendation Network
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
        shortterm_hu = torch.zeros((len(u), self.hidden_size)).float().to(rui.device)
        shortterm_hi = torch.zeros((len(i), self.hidden_size)).float().to(rui.device)
        
        return shortterm_hu, shortterm_hi
        
        
    def forward(self, u_emb, i_emb, edge_index, rui, riu):
        # propagate information
        # longterm
        hLu, hLi = self.longterm(u_emb, i_emb, edge_index, rui, riu)
        
        # shortterm
        hSu, hSi = self.shortterm(u_emb, i_emb, edge_index, rui, riu)
        
        return hLu, hSu, hLi, hSi