import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm

"""
Based on https://github.com/dwromero/ckconv/blob/master/ckconv/nn/ckconv.py
"""

# From LieConv
class Expression(torch.nn.Module):
    def __init__(self, func):
        """
        Creates a torch.nn.Module that applies the function func.
        :param func: lambda function
        """
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def Multiply(
    omega_0: float,
):
    """
    out = omega_0 * x
    """
    return Expression(lambda x: omega_0 * x)


def Sine():
    """
    out = sin(x)
    """
    return Expression(lambda x: torch.sin(x))

class SirenKernel(nn.Module): # Siren Kernel
    def __init__(self,
                in_size = 1,
                out_size = 64*64,
                layer_sizes = [32, 64],
                omega_0 = 30,
                bias = False
                ):
        super().__init__()

        # settings
        self.omega_0 = omega_0
        self.sizes = [in_size] + layer_sizes + [out_size]

        # network
        self.layers = nn.ModuleList()

        for i in self.sizes[:-2]:
            self.layers.append(weight_norm(nn.Linear(self.sizes[i], self.sizes[i+1], bias=bias)))
            self.layers.append(Multiply(self.omega_0))
            # TODO: BatchNorm1D ?
            self.layers.append(Sine)

        self.layers.append(weight_norm(nn.Linear(self.sizes[-2], self.sizes[-1], bias=bias)))
        # TODO: Dropout ?


    def forward(self, t):
        x = t
        for layer in self.layers:
            x = layer(x)

        return x
    



# TODO: random fourier kernel?