import collections
import torch
from torch import nn

from .misc import Conv2d, LeakyReLU


class SubPixelConv(nn.Module):
    """
    Rearranges elements in a tensor of shape :math:`(B, C \\times r^2, H, W)` to a tensor of shape
    :math:`(B, C, H \\times r, W \\times r)`.

    Look at the paper: `Real-Time Single Image and Video Super-Resolution
    Using an Efficient Sub-Pixel Convolutional Neural Network`_
    for more details.

    Args:
        num_features: Number of channels in the input tensor.
        scale_factor: Factor to increase spatial resolution by.

    .. _`Real-Time Single Image and Video Super-Resolution Using an Efficient
        Sub-Pixel Convolutional Neural Network`:
        https://arxiv.org/pdf/1609.05158.pdf

    """

    def __init__(self, num_features: int, scale_factor: int = 2):
        super().__init__()

        # Check that scale factor is a power of two (2^n)
        if (scale_factor & (scale_factor - 1)) != 0:
            raise ValueError(f'scale {scale_factor} is not supported. Supported scale is: 2^n.')

        # Define upscaling block
        self.block = nn.Sequential(collections.OrderedDict([
            ("conv", Conv2d(num_features, num_features * 4)),
            ("px_shuffle", nn.PixelShuffle(upscale_factor=scale_factor)),
            ("act", LeakyReLU()),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Apply conv -> shuffle pixels -> apply non-linearity.

        Args:
            x: Batch of inputs.

        Returns:
            Upscaled input.

        """
        return self.block(x)
