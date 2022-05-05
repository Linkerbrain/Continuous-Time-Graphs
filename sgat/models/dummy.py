import time

import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from sgat.models.sgat_module import SgatModule


class Dummy(SgatModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(1,1)


    @staticmethod
    def add_args(parser):
        parser.add_argument('--mode', type=str, options=['dumb', 'neighbour_cheat'])

    def forward(self, graph, predict_u, predict_i, predict_i_ptr=True):
        p = torch.zeros_like(predict_u, device=self.device, dtype=torch.float)
        return torch.sigmoid(self.linear(p.unsqueeze(1)).squeeze())
