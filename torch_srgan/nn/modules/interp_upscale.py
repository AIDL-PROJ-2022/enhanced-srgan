"""
Interpolation up-scaling module.
"""

__author__ = "Marc Bermejo"


import collections
import torch
from torch import nn

from .misc import Conv2dK3, LeakyReLUSlopeDot2


class InterpUpscale(nn.Module):
    """
    Up-samples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    Shape:
        - Input: :math:`(N, C, W_{in})`, :math:`(N, C, H_{in}, W_{in})` or :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, W_{out})`, :math:`(N, C, H_{out}, W_{out})`
          or :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

    .. math::
        D_{out} = \left\lfloor D_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

    Args:
        num_features: Number of channels in the input tensor.
        scale_factor: Factor to increase spatial resolution by.
        upsample_mode: the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
    """

    def __init__(self, num_features: int, scale_factor: int = 2, upsample_mode: str = 'nearest'):
        super(InterpUpscale, self).__init__()

        # Define upscaling block
        self.block = nn.Sequential(collections.OrderedDict([
            ("interp", nn.Upsample(scale_factor=scale_factor, mode=upsample_mode)),
            ("conv", Conv2dK3(num_features, num_features)),
            ("act", LeakyReLUSlopeDot2()),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Apply interpolation -> conv -> non-linearity.

        Args:
            x: Batch of inputs.

        Returns:
            Upscaled input.

        """
        return self.block(x)
