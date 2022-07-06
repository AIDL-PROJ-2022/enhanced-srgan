import collections
import torch.nn as nn

from typing import List, Tuple

from ..nn.modules import Conv2d, LeakyReLU


class VGGStyleDiscriminator(nn.Module):
    """

    """
    def __init__(self, img_channels: int = 3, vgg_blk_ch: Tuple[int] = (64, 64, 128, 128, 256, 256, 512, 512),
                 fc_features: Tuple[int] = (1024, )):
        super(VGGStyleDiscriminator, self).__init__()

        # Initialize convolutional blocks array
        cnn_blocks: List[Tuple[str, nn.Module]] = []

        # Set the discriminator input convolution block and append it to blocks list
        first_conv = collections.OrderedDict([
            ("conv", Conv2d(img_channels, vgg_blk_ch[0], stride=1, bias=True)),
            ("act", LeakyReLU()),
        ])
        cnn_blocks.append((f"block_0", nn.Sequential(first_conv)))

        # Initialize latent space input channels to first block output channels
        in_ch = vgg_blk_ch[0]
        # Generate VGG-like blocks of the discriminator
        for i, out_ch in enumerate(vgg_blk_ch[1:], start=1):
            # Calculate the stride for this convolutional block. If the convolutional layer has the same input and
            # output channels, we need to apply a stride of two, that is equivalent of doing: 'conv + 2x2 pooling'.
            stride = 2 if in_ch == out_ch else 1
            # Create convolutional block
            block_list = collections.OrderedDict([
                ("conv", Conv2d(in_ch, out_ch, stride=stride, bias=False)),
                ("bn", nn.BatchNorm2d(out_ch)),
                ("act", LeakyReLU()),
            ])
            block = nn.Sequential(collections.OrderedDict(block_list))
            cnn_blocks.append((f"block_{i}", block))
            # Update next convolutional block input channels
            in_ch = out_ch
        # Add an adaptive average pooling layer as the last feature extractor block to ensure a fixed input
        # dimension to the fully connected classifier head.
        cnn_blocks.append(("avg_pool", nn.AdaptiveAvgPool2d((7, 7))))

        # Convert convolutional blocks list to a PyTorch sequence
        self.features = nn.Sequential(collections.OrderedDict(cnn_blocks))

        # Initialize fully connected blocks array
        fc_blocks: List[Tuple[str, nn.Module]] = []

        # Initialize fully connected layers input feature size from the last convolution channels and output size
        fc_in_feat = int(vgg_blk_ch[-1] * 7 * 7)
        # Generate the fully connected layer blocks
        for i, fc_out_feat in enumerate(fc_features):
            # Create fully connected block
            block_list = collections.OrderedDict([
                ("linear", nn.Linear(fc_in_feat, fc_out_feat)),
                ("act", LeakyReLU()),
            ])
            block = nn.Sequential(collections.OrderedDict(block_list))
            fc_blocks.append((f"block_{i}", block))
            # Update next fully connected block input features
            fc_in_feat = fc_out_feat
        # Add last fully connected layer with an output size of one
        fc_blocks.append((f"linear_out", nn.Linear(fc_in_feat, 1)))

        # Convert fully connected blocks list to a PyTorch sequence
        self.head = nn.Sequential(collections.OrderedDict(fc_blocks))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Batch of inputs.

        Returns:
            Processed batch.
        """
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)

        return x
