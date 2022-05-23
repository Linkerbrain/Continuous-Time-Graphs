import torch
from torch import nn
import numpy as np

from ctgraph.models.recommendation.module import RecommendationModule

from .dgsr_utils import sparse_dense_mul, pass_messages, relative_order, get_last

from ctgraph import logger
from siren_pytorch import SirenNet
from siren_pytorch import Sine


class Ct(nn.Module):
    @staticmethod
    def add_args(parser):
        # layer settings
        parser.add_argument('--num_siren_layers', type=int, default=5)
        parser.add_argument('--dim_hidden_size', type=int, default=256)
        parser.add_argument('--w0_init', type=int, default=30)

    def __init__(self,user_or_item, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_siren_layers = self.params.num_siren_layers
        self.input_size = self.params.input_size
        self.embedding_size = self.params.embedding_size
        self.w0_init = self.params.w0_init
        self.dim_hidden_size = self.params.dim_hidden_size

        # enable True for user and False for item
        self.user_or_item = user_or_item

        self.net = SirenNet(
            dim_in=self.input_size,  # input dimension
            dim_hidden=self.params.dim_hidden_size,  # hidden dimension
            dim_out=self.input_size * self.embedding_size,  # output dimension
            num_layers=self.num_siren_layers,  # number of layers
            final_activation=nn.Sigmoid(),  # activation of final layer (nn.Identity() for direct output)
            w0_initial=self.w0_init
            # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )


    def forward(self, batch, edges):
        # initialize user or items
        if self.user_or_item:
            code = 'u'
            index = 0
        else:
            code = 'i'
            index = 1

        # calculate difference of t
        max_t = batch[code].t_max[batch[('u', 'b', 'i')].edge_index[index]]
        t_diff = max_t - batch[code].t
        output = self.net(t_diff)
        return output