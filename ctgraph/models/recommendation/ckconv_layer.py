import torch
import torch.nn as nn
import numpy as np

from .dgsr_utils import sparse_dense_mul, pass_messages, relative_order, get_last

from .ckconv_kernel import SirenLibraryKernel, SirenCustomKernel
from .ckconv_kernel2 import KernelNet
from ctgraph.stats import ParameterizedDistribution as PD

class CKConv(nn.Module): # Continuous Kernel Convolution
    def __init__(self, params):
        super().__init__()

        self.hidden_size = params.embedding_size
        self.td_correction = params.td_correction

        if self.td_correction:
            self.pd = PD.load(params)
        
        # Kernel settings
        in_size = 1
        out_size = params.embedding_size*params.embedding_size

        kernel_hidden_size = 50
        omega_0 = 30
        bias = True

        dropout = 0.3

        # using Siren Library
        # self.w_items = SirenLibraryKernel(in_size, out_size, layer_sizes, omega_0, bias)
        # self.w_users = SirenLibraryKernel(in_size, out_size, layer_sizes, omega_0, bias)

        self.w_items = KernelNet(in_size, out_size, kernel_hidden_size, 'Sine', 'LayerNorm', 1, bias, omega_0, dropout)
        self.w_users = KernelNet(in_size, out_size, kernel_hidden_size, 'Sine', 'LayerNorm', 1, bias, omega_0, dropout)


    def forward(self, u_embedded, i_embedded, user_per_trans, item_per_trans, edges_t, u_t, i_t):
        # make times relative
        relative_i = u_t[user_per_trans] - edges_t
        relative_u = i_t[item_per_trans] - edges_t

        if self.td_correction:
            propensities = 1/self.pd.forward(edges_t)
            propensities /= torch.sum(propensities)
            propensities = propensities[:, None, None]
        else:
            propensities = 1

        # get kernels
        user_kernels = self.w_users(relative_u.view(-1, 1, 1)).view((len(edges_t), self.hidden_size, self.hidden_size))
        item_kernels = self.w_items(relative_i.view(-1, 1, 1)).view((len(edges_t), self.hidden_size, self.hidden_size))

        # hLu_counts = torch.zeros_like(u_embedded)
        # hLu_counts.index_add_(0, user_per_trans, torch.ones_like(item_per_trans, dtype=torch.float))

        # hLi_counts = torch.zeros_like(i_embedded)
        # hLi_counts.index_add_(0, item_per_trans, torch.ones_like(user_per_trans, dtype=torch.float))

        # propagate item messages to user embeddings
        item_messages = ((item_kernels * propensities) @ i_embedded[item_per_trans].unsqueeze(-1)).squeeze()
        item_messages_normalized = item_messages #/ hLi_counts[item_per_trans].sqrt()

        hLu = torch.zeros_like(u_embedded)
        hLu.index_add_(0, user_per_trans, item_messages_normalized)

        hLu_normalized = hLu #/ hLu_counts

        # propagate user messages to item embeddings
        user_messages = ((user_kernels * propensities) @ u_embedded[user_per_trans].unsqueeze(-1)).squeeze()
        user_messages_normalized = user_messages #/ hLu_counts[user_per_trans].sqrt()


        hLi = torch.zeros_like(i_embedded)
        hLi.index_add_(0, item_per_trans, user_messages_normalized)

        hLi_normalized = hLi #/ hLi_counts

        # import pdb; pdb.set_trace()
        return hLu_normalized, hLi_normalized
