import numpy_indexed as npi
import torch
from torch import nn

from ctgraph.models.recommendation.module import RecommendationModule


class Dummy(RecommendationModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(1,1)


    @staticmethod
    def add_args(parser):
        parser.add_argument('--mode', type=str, choices=['dumb', 'cheat'], default='dumb')

    def forward(self, graph, predict_u, predict_i, predict_i_ptr=True):
        # Convert codes to indices (ptr's) if necessary
        if predict_i_ptr:
            predict_i = graph['i'].code[predict_i]

        p = torch.zeros_like(predict_u, device=self.device, dtype=torch.float)
        p = torch.sigmoid(self.linear(p.unsqueeze(1)).squeeze())
        if self.params.mode == 'cheat':
            s = graph['u', 's', 'i'].edge_index.cpu().numpy()
            labels = graph['u', 's', 'i'].label.bool().cpu().numpy()
            t = s[:, labels]
            t[1] = graph['i'].code[t[1]]
            r = torch.stack([predict_u, predict_i], dim=0).cpu().numpy()
            p = torch.relu(p) * torch.tensor(npi.contains(t, r, axis=1)).float()
        return p
