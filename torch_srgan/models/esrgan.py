import collections
import torch
import torch.nn as nn
import numpy as np

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


# class DiscriminatorESRGAN(nn.Module):
#     """
#
#     """
#     def __init__(self, img_channels: int = 3, vgg_blk_ch: Tuple[int] = (64, 64, 128, 128, 256, 256, 512, 512),
#                  fc_features: Tuple[int] = (1024, )):
#         super(DiscriminatorESRGAN, self).__init__()
#
#         # Initialize convolutional blocks array
#         cnn_blocks: List[Tuple[str, nn.Module]] = []
#
#         # Set the discriminator input convolution block and append it to blocks list
#         first_conv = collections.OrderedDict([
#             ("conv", Conv2d(img_channels, vgg_blk_ch[0], stride=1)),
#             ("act", LeakyReLU()),
#         ])
#         cnn_blocks.append((f"block_0", nn.Sequential(first_conv)))
#
#         # Initialize latent space input channels to first block output channels
#         in_ch = vgg_blk_ch[0]
#         # Generate VGG-like blocks of the discriminator
#         for i, out_ch in enumerate(vgg_blk_ch[1:], start=1):
#             # Calculate the stride for this convolutional block. If the convolutional layer has the same input and
#             # output channels, we need to apply a stride of two, that is equivalent of doing: 'conv + 2x2 pooling'.
#             stride = 2 if in_ch == out_ch else 1
#             # Create convolutional block
#             block_list = collections.OrderedDict([
#                 ("conv", Conv2d(in_ch, out_ch, stride=stride)),
#                 ("bn", nn.BatchNorm2d(out_ch)),
#                 ("act", LeakyReLU()),
#             ])
#             block = nn.Sequential(collections.OrderedDict(block_list))
#             cnn_blocks.append((f"block_{i}", block))
#             # Update next convolutional block input channels
#             in_ch = out_ch
#         # Add an adaptive average pooling layer as the last feature extractor block to ensure a fixed input
#         # dimension to the fully connected classifier head.
#         cnn_blocks.append(("avg_pool", nn.AdaptiveAvgPool2d((7, 7))))
#
#         # Convert convolutional blocks list to a PyTorch sequence
#         self.features = nn.Sequential(collections.OrderedDict(cnn_blocks))
#
#         # Initialize fully connected blocks array
#         fc_blocks: List[Tuple[str, nn.Module]] = []
#
#         # Initialize fully connected layers input feature size from the last convolution channels and output size
#         fc_in_feat = int(vgg_blk_ch[-1] * 7 * 7)
#         # Generate the fully connected layer blocks
#         for i, fc_out_feat in enumerate(fc_features):
#             # Create fully connected block
#             block_list = collections.OrderedDict([
#                 ("linear", nn.Linear(fc_in_feat, fc_out_feat)),
#                 ("act", LeakyReLU()),
#             ])
#             block = nn.Sequential(collections.OrderedDict(block_list))
#             fc_blocks.append((f"block_{i}", block))
#             # Update next fully connected block input features
#             fc_in_feat = fc_out_feat
#         # Add last fully connected layer with an output size of one
#         fc_blocks.append((f"linear_out", nn.Linear(fc_in_feat, 1)))
#
#         # Convert fully connected blocks list to a PyTorch sequence
#         self.head = nn.Sequential(collections.OrderedDict(fc_blocks))
#
#     def forward(self, x):
#         """
#         Forward pass.
#
#         Args:
#             x: Batch of inputs.
#
#         Returns:
#             Processed batch.
#         """
#         x = self.features(x)
#         x = x.view(x.shape[0], -1)
#         x = self.head(x)
#
#         return x

class DiscriminatorESRGAN(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super(DiscriminatorESRGAN, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out
