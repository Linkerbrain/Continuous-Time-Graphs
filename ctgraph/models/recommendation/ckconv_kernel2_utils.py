import torch


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


class MultiplyLearned(torch.nn.Module):
    def __init__(
        self,
        omega_0: float,
    ):
        """
        out = omega_0 * x, with a learned omega_0
        """
        super().__init__()
        self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
        with torch.no_grad():
            self.omega_0.fill_(omega_0)

    def forward(self, x):
        return 100 * self.omega_0 * x


import torch.nn as nn


def Linear1d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
):
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def Linear2d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True,
):
    """
    Implements a Linear Layer in terms of a point-wise convolution.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

class LayerNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-12,
    ):
        """Uses GroupNorm implementation with group=1 for speed."""
        super().__init__()
        # we use GroupNorm to implement this efficiently and fast.
        self.layer_norm = torch.nn.GroupNorm(1, num_channels=num_channels, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)

def Swish():
    """
    out = x * sigmoid(x)
    """
    return Expression(lambda x: x * torch.sigmoid(x))


def Sine():
    """
    out = sin(x)
    """
    return Expression(lambda x: torch.sin(x))

import torch.nn.functional as f
import torch.fft

from typing import Tuple, Optional


def causal_padding(
    x: torch.Tensor,
    kernel: torch.Tensor,
):
    # 1. Pad the input signal & kernel tensors.
    # Check if sizes are odd. If not, add a pad of zero to make them odd.
    if kernel.shape[-1] % 2 == 0:
        kernel = f.pad(kernel, [1, 0], value=0.0)
        # x = torch.nn.functional.pad(x, [1, 0], value=0.0)
    # 2. Perform padding on the input so that output equals input in length
    x = f.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)

    return x, kernel


def causal_conv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """

    x, kernel = causal_padding(x, kernel)
    return torch.nn.functional.conv1d(x, kernel, bias=bias, padding=0)


def causal_fftconv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    double_precision: bool = False,
):
    """
    Args:
        x: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """

    x_shape = x.shape
    # 1. Handle padding of the input and the kernel to make them odd.
    x, kernel = causal_padding(x, kernel)

    # 2. Pad the kernel tensor to make them equally big. Required for fft.
    kernel = f.pad(kernel, [0, x.size(-1) - kernel.size(-1)])

    # 3. Perform fourier transform
    if double_precision:
        # We can make usage of double precision to make more accurate approximations of the convolution response.
        x = x.double()
        kernel = kernel.double()

    x_fr = torch.fft.rfft(x, dim=-1)
    kernel_fr = torch.fft.rfft(kernel, dim=-1)

    # 4. Multiply the transformed matrices:
    # (Input * Conj(Kernel)) = Correlation(Input, Kernel)
    kernel_fr = torch.conj(kernel_fr)
    output_fr = (x_fr.unsqueeze(1) * kernel_fr.unsqueeze(0)).sum(
        2
    )  # 'ab..., cb... -> ac...'

    # 5. Compute inverse FFT, and remove extra padded values
    # Once we are back in the spatial domain, we can go back to float precision, if double used.
    out = torch.fft.irfft(output_fr, dim=-1).float()

    out = out[:, :, : x_shape[-1]]

    # 6. Optionally, add a bias term before returning.
    if bias is not None:
        out = out + bias.view(1, -1, 1)

    return out