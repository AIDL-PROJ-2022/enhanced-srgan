import collections
import torch.nn as nn

from typing import List, Tuple

from ..nn.modules import Conv2d, LeakyReLU, ResidualInResidualDenseBlock, SubPixelConv, InterpUpscale


class RRDBNet(nn.Module):
    """'
    RRDB network, processing images in LR space and scaling them to HR space.

    It has been proposed in `ESRGAN: Enhanced Super-Resolution
    Generative Adversarial Networks`_.

    Args:
        img_channels: Number of channels in the input and output image.
        scale_factor: Ratio between the size of the high-resolution image (output) and
            its low-resolution counterpart (input).
        rrdb_channels: Number of channels produced by the Residual-in-Residual Dense blocks.
        growth_channels: Number of channels in the Residual-in-Residual Dense block latent space.
        num_basic_blocks: Number of basic `RRDB` blocks to use.
        num_dense_blocks: Number of dense blocks to use to form `RRDB` block.
        num_residual_blocks: Number of convolutions to use to form dense block.
        residual_scaling: Residual connections scaling factor.

    .. _`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks`:
        https://arxiv.org/pdf/1809.00219.pdf

    """
    def __init__(self, img_channels: int = 3, scale_factor: int = 2, rrdb_channels: int = 64,
                 growth_channels: int = 32, num_basic_blocks: int = 23, num_dense_blocks: int = 3,
                 num_residual_blocks: int = 5, residual_scaling: float = 0.2, use_subpixel_conv: bool = False):
        super(RRDBNet, self).__init__()

        # Check that scale factor is a power of two (2^n)
        if (scale_factor & (scale_factor - 1)) != 0:
            raise ValueError(f'scale {scale_factor} is not supported. Supported scale is: 2^n.')

        # First convolutional layer. This layer creates needed channels for RRDB blocks.
        self.in_conv = Conv2d(img_channels, rrdb_channels)

        rrdb_blocks_list: List[Tuple[str, nn.Module]] = []
        # Define Residual-in-Residual Dense blocks
        for i in range(num_basic_blocks):
            rrdb_block = ResidualInResidualDenseBlock(
                num_features=rrdb_channels, growth_channels=growth_channels, num_dense_blocks=num_dense_blocks,
                num_blocks=num_residual_blocks, residual_scaling=residual_scaling,
            )
            rrdb_blocks_list.append((f"rrdb_{i}", rrdb_block))
        # Define Residual-in-Residual Dense blocks output convolution
        rrdb_blocks_list.append((f"out_conv", Conv2d(rrdb_channels, rrdb_channels)))
        # Convert RRDB blocks list to a PyTorch sequence
        self.rrdb_blocks = nn.Sequential(collections.OrderedDict(rrdb_blocks_list))

        upsampling_blocks_list: List[Tuple[str, nn.Module]] = []
        # Define upsampling blocks
        for i in range(scale_factor // 2):
            if use_subpixel_conv:
                upsampling_block = SubPixelConv(num_features=rrdb_channels, scale_factor=2)
            else:
                upsampling_block = InterpUpscale(num_features=rrdb_channels, scale_factor=2)
            upsampling_blocks_list.append((f"upsampling_{i}", upsampling_block))
        # Define upsampling last convolution block
        last_conv = nn.Sequential(
            Conv2d(rrdb_channels, rrdb_channels),
            LeakyReLU(),
            Conv2d(rrdb_channels, img_channels),
        )
        upsampling_blocks_list.append(("out_conv", last_conv))
        # Convert upsampling blocks list to a PyTorch sequence
        self.upsampling_blocks = nn.Sequential(collections.OrderedDict(upsampling_blocks_list))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Batch of inputs.

        Returns:
            Processed batch.
        """
        rrdb_in = self.in_conv(x)
        out = self.rrdb_blocks(rrdb_in)
        out += rrdb_in
        out = self.upsampling_blocks(out)

        return out
