import time

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from sgat.models import SgatModule


class DGSR(SgatModule):
    
    @staticmethod
    def add_args(parser):
        pass
    
    def forward(self, graph, predict_u, predict_i):
        pass