import torch
import torch.nn as nn
import numpy as np

from .dgsr_utils import sparse_dense_mul, pass_messages, relative_order, get_last

class SumConv(nn.Module): # Continuous Kernel Convolution
    def __init__(self
                ):
        super().__init__()
        bias = False


    def forward(self, u_embedded, i_embedded, user_per_trans, item_per_trans, edges_t, u_t, i_t):
        hLu = torch.zeros_like(u_embedded)
        hLu.index_add_(0, user_per_trans, i_embedded[item_per_trans])

        hLi = torch.zeros_like(i_embedded)
        hLi.index_add_(0, item_per_trans, u_embedded[user_per_trans])

        return hLu, hLi
