import torch
import torch.nn as nn
import numpy as np

from .dgsr_utils import sparse_dense_mul, pass_messages, relative_order, get_last

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DGSRLayer(nn.Module): # Dynamic Graph Recommendation Network
    def __init__(self,
                 user_num, item_num,
                 hidden_size,
                 user_max, item_max,
                 shortterm,
                 edge_attr,
                 params
                ):
        super().__init__()
        """ init """
        self.params = params

        self.user_vocab_num = user_num
        self.item_vocab_num = item_num

        self.user_max = user_max
        self.item_max = item_max

        self.hidden_size = hidden_size
        self.sqrt_d = np.sqrt(self.hidden_size)

        self.do_shortterm = shortterm

        self.edge_attr = edge_attr

        """ layers """
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Long Term Attention Item
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Long Term Attention User

        self.w1b = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Long Term Item
        self.w2b = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Long Term User

        if self.do_shortterm:
            self.w3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Short Term Item
            self.w4 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Short Term User

    def longterm(self, u_embedded, i_embedded, edge_index, pVui, pKiu):
        # --- long term ---

        user_messages_for_attention = self.w2(u_embedded) # (u, h)
        item_messages_for_attention = self.w1(i_embedded) # (i, h)

        # message similarity
        e = (user_messages_for_attention) @ (item_messages_for_attention).T # (u, i)
        e = sparse_dense_mul(edge_index, e) # (u, i)

        user_per_trans, item_per_trans = edge_index.indices()

        # - users to items -

        # compute positional embeddings
        if pVui is not None:
            # dot product van elke pos embedding met betreffende user
            u_at_pVui = torch.einsum('ij, ij->i', user_messages_for_attention[user_per_trans], pVui)

            e_ui_values = e._values() + u_at_pVui
        else:
            e_ui_values = e._values()

        # alpha is softmax(wu @ wi.T + wu @ p)
        e_ui = torch.sparse_coo_tensor(e._indices(), e_ui_values, e.size())
        alphas = torch.sparse.softmax(e_ui / self.sqrt_d, dim=1) # (u, i)

        # - items to users -
        
        if pKiu is not None:
            # dot product van elke pos embedding met betreffende user
            u_at_pKiu = torch.einsum('ij, ij->i', item_messages_for_attention[item_per_trans], pKiu)

            e_iu_values = e._values() + u_at_pKiu
        else:
            e_iu_values = e._values()

        # beta is softmax(wi @ wu.T + wi @ p)
        e_trans = torch.transpose(e, 0, 1)
        e_iu = torch.sparse_coo_tensor(e_trans._indices(), e_iu_values, e_trans.size())
        betas = torch.sparse.softmax(e_iu / self.sqrt_d, dim=1) # (u, i)

        # pass messages
        user_messages_b = self.w2b(u_embedded) # (u, h)
        item_messages_b = self.w1b(i_embedded) # (i, h)

        longterm_hu = pass_messages(item_messages_b, alphas, pKiu)
        longterm_hi = pass_messages(user_messages_b, betas, pVui)

        return longterm_hu, longterm_hi

    def shortterm(self, u_embedded, i_embedded, edge_index, graph, last_u, last_i):
        # --- short term ---
        user_per_trans, item_per_trans = edge_index.indices()

        user_messages = self.w2(u_embedded) # (u, h)
        item_messages = self.w1(i_embedded) # (i, h)

        # Get last item
        # last_item = get_last(user_messages.device, user_per_trans, item_per_trans, graph['i'].code).to(torch.int)
        last_item = last_u[1]
        last_item_embedding = i_embedded[last_item]
        last_item = self.w3(last_item_embedding)

        # Get last user from items
        # last_user = get_last(item_messages.device, item_per_trans, user_per_trans, graph['u'].code).to(torch.int)
        last_user = last_i[1]
        last_user_embedding = u_embedded[last_user]
        last_user = self.w4(last_user_embedding)

        # message similarity alpha
        a = (last_item) @ (item_messages).T # (u, i)
        a = sparse_dense_mul(edge_index, a) # (u, i)

        # message similarity beta
        b = (last_user) @ (item_messages).T # (u, i)
        b = sparse_dense_mul(edge_index, b) # (u, i)


        # compute alphas
        a = torch.sparse_coo_tensor(a._indices(), a._values(), a.size())
        alphas = torch.sparse.softmax(a / self.sqrt_d, dim=1) # (u, i)


        # compute betas
        b_trans = torch.transpose(b, 0, 1)
        b = torch.sparse_coo_tensor(b_trans._indices(), b_trans._values(), b_trans.size())
        betas = torch.sparse.softmax(b / self.sqrt_d, dim=1) # (u, i)


        # pass messages
        shortterm_hu = pass_messages(item_messages, alphas, torch.ones((alphas._indices().shape[1], item_messages.shape[1])).to(item_messages.device))
        shortterm_hi = pass_messages(user_messages, betas, torch.ones((betas._indices().shape[1], user_messages.shape[1])).to(user_messages.device))


        return shortterm_hu, shortterm_hi


    def forward(self, u_emb, i_emb, edge_index, pVui, pKiu, graph, last_u, last_i):
        # propagate information
        # longterm
        hLu, hLi = self.longterm(u_emb, i_emb, edge_index, pVui, pKiu)

        if self.do_shortterm:
            hSu, hSi = self.shortterm(u_emb, i_emb, edge_index, graph, last_u, last_i)
        else:
            hSu, hSi = None, None

        return hLu, hSu, hLi, hSi
