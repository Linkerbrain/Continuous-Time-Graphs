import torch
from torch import nn
import numpy as np

from ctgraph.models.recommendation.module import RecommendationModule

from .dgsr_utils import sparse_dense_mul, pass_messages, relative_order, get_last

from ctgraph import logger
from siren_pytorch import SirenNet
from siren_pytorch import Sine


class ContinuousTimeEmbedder(nn.Module):
    @staticmethod
    def add_args(parser):
        # layer settings
        parser.add_argument('--num_siren_layers', type=int, default=5)
        parser.add_argument('--dim_hidden_size', type=int, default=256)
        parser.add_argument('--w0_init', type=int, default=30)

    def __init__(self, for_users, params, mode='t_max'):
        super().__init__()
        self.num_siren_layers = params.num_siren_layers
        self.embedding_size = params.embedding_size
        self.w0_init = params.w0_init
        self.dim_hidden_size = params.dim_hidden_size

        # enable True for user and False for item
        self.for_users = for_users
        self.mode = mode

        self.net = SirenNet(
            dim_in=1,  # input dimension
            dim_hidden=params.dim_hidden_size,  # hidden dimension
            dim_out=self.embedding_size,  # output dimension
            num_layers=self.num_siren_layers,  # number of layers
            final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
            w0_initial=self.w0_init
            # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )


    def forward(self, batch):
        # initialize user or items
        if self.for_users:
            code = 'u'
            index = 0
        else:
            code = 'i'
            index = 1

        if self.mode == 't_min':
            # calculate difference of t
            max_t = batch[code].t_max[batch[('u', 'b', 'i')].edge_index[index]]
            t_diff = max_t - batch['u', 'b', 'i'].t
            output = self.net(t_diff.reshape((-1, 1)))
        elif self.mode == 't_max':
            # calculate difference of t
            min_t = batch[code].t_min[batch[('u', 'b', 'i')].edge_index[index]]
            t_diff = min_t - batch['u', 'b', 'i'].t
            output = self.net(t_diff.reshape((-1, 1)))
        elif self.mode == 'absolute':
            output = self.net(batch['u', 'b', 'i'].t)
        else:
            raise NotImplementedError

        return output