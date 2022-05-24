import torch
import torch.nn as nn
import numpy as np

from .dgsr_utils import sparse_dense_mul, pass_messages, relative_order, get_last

from .ckconv_kernel import SirenLibraryKernel, SirenCustomKernel

class CKConv(nn.Module): # Continuous Kernel Convolution
    def __init__(self,
                 hidden_size
                ):
        super().__init__()

        self.hidden_size = hidden_size
        
        # Kernel settings
        in_size = 1
        out_size = hidden_size*hidden_size
        layer_sizes = [hidden_size, hidden_size]
        omega_0 = 30
        bias = False

        self.w_items = SirenLibraryKernel(in_size, out_size, layer_sizes, omega_0, bias)
        self.w_users = SirenLibraryKernel(in_size, out_size, layer_sizes, omega_0, bias)


    def forward(self, u_embedded, i_embedded, user_per_trans, item_per_trans, edges_t, u_t, i_t):
        # make times relative
        relative_u = u_t[user_per_trans] - edges_t
        relative_i = i_t[item_per_trans] - edges_t

        # get kernels
        user_kernels = self.w_users(relative_u).reshape((len(edges_t), self.hidden_size, self.hidden_size))
        item_kernels = self.w_items(relative_i).reshape((len(edges_t), self.hidden_size, self.hidden_size))

        # propagate item messages to user embeddings
        item_messages = (item_kernels @ i_embedded[item_per_trans].unsqueeze(-1)).squeeze()

        hLu = torch.zeros_like(u_embedded)
        hLu.index_add_(0, user_per_trans, item_messages)

        # propagate user messages to item embeddings
        user_messages = (user_kernels @ u_embedded[user_per_trans].unsqueeze(-1)).squeeze()

        hLi = torch.zeros_like(i_embedded)
        hLi.index_add_(0, item_per_trans, user_messages)

        return hLu, hLi
