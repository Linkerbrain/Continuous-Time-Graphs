import torch
import torch.nn as nn
import numpy as np

from .dgsr_utils import sparse_dense_mul, pass_messages, relative_order, get_last

from .ckconv_kernel import SirenKernel

class CKConv(nn.Module): # Continuous Kernel Convolution
    def __init__(self,
                 hidden_size
                ):
        super().__init__()

        self.hidden_size = hidden_size
        
        self.siren_items
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # User Messages



    def forward(self, u_embedded, i_embedded, edges, edges_t):

        hLu = self.w2(u_embedded)
        hLi = self.w1(i_embedded)

        return hLu, hLi


