import torch
from torch import nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    """
    TODO
    """

    def __init__(self, scale_factor: int = 2, mode='nearest'):
        super(Interpolate, self).__init__()

        # Check that scale factor is a power of two (2^n)
        if (scale_factor & (scale_factor - 1)) != 0:
            raise ValueError(f'scale {scale_factor} is not supported. Supported scale is: 2^n.')

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Apply image interpolation to upscale it.

        Args:
            x: Batch of inputs.

        Returns:
            Upscaled input.

        """
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
