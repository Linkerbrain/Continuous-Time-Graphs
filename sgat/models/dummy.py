import time

import numpy_indexed as npi
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from sgat.models.sgat_module import SgatModule


class Dummy(SgatModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.graph
        self.linear = nn.Linear(1,1)


    @staticmethod
    def add_args(parser):
        parser.add_argument('--mode', type=str, choices=['dumb', 'neighbour_cheat'], default='dumb')

    def forward(self, graph, predict_u, predict_i, predict_i_ptr=True):
        p = torch.zeros_like(predict_u, device=self.device, dtype=torch.float)
        p = torch.sigmoid(self.linear(p.unsqueeze(1)).squeeze())
        if self.params.mode == 'neighbour_cheat':
            s = graph['u', 's', 'i'].edge_index.cpu().numpy()
            labels = graph['u', 's', 'i'].label.bool().cpu().numpy()
            t = s[:, labels]
            r = torch.stack([predict_u, predict_i], dim=0).cpu().numpy()
            p = torch.relu(p) * torch.tensor(npi.contains(t, r, axis=1)).float()
        return p
