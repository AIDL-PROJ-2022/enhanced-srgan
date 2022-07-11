"""
Residual-in-residual dense block module.
"""

__author__ = "Marc Bermejo"


import collections
import torch.nn as nn
import torch

from typing import List, Tuple

from .misc import Conv2dK3, LeakyReLUSlopeDot2


class ResidualDenseBlock(nn.Module):
    """
    Basic block of :py:class:`ResidualInResidualDenseBlock`.

    Args:
        num_features: :math:`C` from an expected input of size :math:`(N, C, H, W)`.
        growth_channels: Number of channels in the latent space.
        num_blocks: Number of convolutional blocks to use to form dense block.
        residual_scaling: Residual connections scaling factor.
    """

    def __init__(self, num_features: int, growth_channels: int, num_blocks: int = 5, residual_scaling: float = 0.2):
        super(ResidualDenseBlock, self).__init__()

        blocks: List[nn.Module] = []

        # Initialize the network CNN input channels to the given number of features
        in_channels = num_features

        # Define the convolutional blocks of the latent space
        for i in range(num_blocks - 1):
            # Define the convolutional block
            block = collections.OrderedDict([
                ("conv", Conv2dK3(in_channels, growth_channels)),
                ("act", LeakyReLUSlopeDot2()),
            ])
            # Append it to the blocks array
            blocks.append(nn.Sequential(block))
            # Calculate input channels of the next's block CNN.
            # As we concatenate the input of each block with its output, we need to increment them by the
            # output channels of the last convolution.
            in_channels += growth_channels

        # Define the last convolutional block. This block don't need an activation function.
        block = collections.OrderedDict([
            ("conv", Conv2dK3(in_channels, num_features)),
        ])
        # Append it to the blocks array
        blocks.append(nn.Sequential(block))

        # Convert blocks array to PyTorch ModuleList object to ensure that all modules are properly registered
        self.blocks = nn.ModuleList(blocks)
        # Set residual scaling parameter
        self.residual_scaling = nn.Parameter(torch.tensor(residual_scaling), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Batch of inputs.

        Returns:
            Processed batch.
        """
        # Initialize CNN block input with the input tensor and output
        block_input = x

        # Iterate over all deep network blocks
        for block in self.blocks[:-1]:
            # Calculate CNN block output
            out = block(block_input)
            # Concatenate last CNN block output to next block's input
            block_input = torch.cat([block_input, out], dim=1)
        # Calculate last CNN block output
        out = self.blocks[-1](block_input)

        # Calculate final output by adding the input skip connection
        return x + self.residual_scaling * out


class ResidualInResidualDenseBlock(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB).

    Look at the paper: `ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks`_ for more details.

    Args:
        num_features: :math:`C` from an expected input of size :math:`(N, C, H, W)`.
        growth_channels: Number of channels in the latent space.
        num_dense_blocks: Number of dense blocks to use to form `RRDB` block.
        residual_scaling: Residual connections scaling factor.
        **kwargs: Dense block params.

    .. _`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks`:
        https://arxiv.org/pdf/1809.00219.pdf

    """

    def __init__(self, num_features: int = 64, growth_channels: int = 32, num_dense_blocks: int = 3,
                 residual_scaling: float = 0.2, **kwargs):
        super(ResidualInResidualDenseBlock, self).__init__()

        # Define all residual dense blocs of the RRDB module
        dense_blocks: List[Tuple[str, nn.Module]] = []
        for i in range(num_dense_blocks):
            block = ResidualDenseBlock(
                num_features=num_features, growth_channels=growth_channels,
                residual_scaling=residual_scaling, **kwargs
            )
            dense_blocks.append((f"rdb_{i}", block))

        # Convert blocks array to a sequential collection of PyTorch modules
        self.dense_blocks = nn.Sequential(collections.OrderedDict(dense_blocks))
        # Set residual scaling parameter
        self.residual_scaling = nn.Parameter(torch.tensor(residual_scaling), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Batch of inputs.

        Returns:
            Processed batch.
        """
        return x + self.residual_scaling * self.dense_blocks(x)

