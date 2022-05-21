import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from siren_pytorch import SirenNet


""" using Siren library """

class SirenLibraryKernel(nn.Module):
    def __init__(self,
                    in_size = 1,
                    out_size = 64*64,
                    layer_sizes = [32, 64],
                    omega_0 = 30,
                    bias = False
                ):
        super().__init__()

        self.siren = SirenNet(
            dim_in = in_size,                        # input dimension, ex. 2d coor
            dim_hidden = layer_sizes[0],         # todo make single parameter                  # hidden dimension
            dim_out = out_size,                       # output dimension, ex. rgb value
            num_layers = len(layer_sizes),                    # number of layers
            final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
            w0_initial = omega_0                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )

    def forward(self, t):
        # give each t their own dim
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # compute kernel
        out = self.siren(t)

        return out


""" utils """

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

""" Kernel """

class SirenCustomKernel(nn.Module):
    """
    Based on ckconv github
    https://github.com/dwromero/ckconv/blob/master/ckconv/nn/ckconv.py
    """
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