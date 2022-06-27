import collections
import torch.nn as nn

from typing import List, Tuple

from ..nn.modules import Conv2d, LeakyReLU


# class VGGStyleDiscriminator(nn.Module):
#     """
#
#     """
#     def __init__(self, img_channels: int = 3, vgg_blk_ch: Tuple[int] = (64, 64, 128, 128, 256, 256, 512, 512),
#                  fc_features: Tuple[int] = (1024, )):
#         super(VGGStyleDiscriminator, self).__init__()
#
#         # Initialize convolutional blocks array
#         cnn_blocks: List[Tuple[str, nn.Module]] = []
#
#         # Set the discriminator input convolution block and append it to blocks list
#         first_conv = collections.OrderedDict([
#             ("conv", Conv2d(img_channels, vgg_blk_ch[0], stride=1, bias=True)),
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
#                 ("conv", Conv2d(in_ch, out_ch, stride=stride, bias=False)),
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

class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 128 or self.input_size == 256, (
            f'input size must be 128 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out
