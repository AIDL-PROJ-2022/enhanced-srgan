import collections
import torch
import torch.nn as nn

from typing import List, Tuple

from ..nn.modules import Conv2d, LeakyReLU, ResidualInResidualDenseBlock, SubPixelConv


class GeneratorESRGAN(nn.Module):
    """'
    Encoder' part of ESRGAN network, processing images in LR space.

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
                 num_residual_blocks: int = 5, residual_scaling: float = 0.2):
        super(GeneratorESRGAN, self).__init__()

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
            upsampling_blocks_list.append((f"upsampling_{i}", SubPixelConv(num_features=rrdb_channels)))
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
        out = torch.clamp(out, min=0.0, max=1.0)
        return out


class DiscriminatorESRGAN(nn.Module):
    def __init__(self, input_shape):
        super(DiscriminatorESRGAN, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)]
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        output = None
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
            output = out_filters

        layers.append(nn.Conv2d(output, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
