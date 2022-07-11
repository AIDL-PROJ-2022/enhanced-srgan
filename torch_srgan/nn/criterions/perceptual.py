"""
VGG perceptual loss implementation.
"""

__author__ = "Marc Bermejo"

import torch

from collections import OrderedDict
from typing import Dict, Tuple

from torch import nn
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg


_VGG_LAYER_NAMES = {
    'vgg11': (
        'conv1_1', 'relu1_1', 'pool1',                          # Block 1
        'conv2_1', 'relu2_1', 'pool2',                          # Block 2
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',    # Block 3
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',    # Block 4
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'     # Block 6
    ),
    'vgg13': (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',    # Block 1
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',    # Block 2
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',    # Block 3
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',    # Block 4
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'     # Block 5
    ),
    'vgg16': (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',                        # Block 1
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',                        # Block 2
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',  # Block 3
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',  # Block 4
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'   # Block 5
    ),
    'vgg19': (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',                                                # Block 1
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',                                                # Block 2
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',    # Block 3
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',    # Block 4
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'     # Block 5
    )
}


def _insert_bn_layers(layer_names: Tuple[str]) -> Tuple[str]:
    """
    Insert bn layer after each conv.

    Args:
        layer_names: The list of layer names.

    Returns:
        The list of layer names with bn layers.
    """
    names_bn = []

    for name in layer_names:
        names_bn.append(name)
        if name.startswith('conv'):
            position = name.replace('conv', '')
            names_bn.append('bn' + position)

    return tuple(names_bn)


class PerceptualLoss(_Loss):
    """
    VGG perceptual loss.
    Calculates loss between features of `model` using a pre-trained VGG network
    for generated and ground-truth images.

    Args:
        layer_weights: The weight for each layer of the VGG network.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4 feature layer
            (before relu5_4) will be used to calculate the loss with a weight of 1.0.
        vgg_type: The type of VGG network used as feature extractor.
        criterion: Loss function to compute distance between features.
        normalize_input: If True, normalize the input image before doing inference though the VGG network.
            The mean and standard deviation values are calculated for an image in the range [0, 1].
        normalize_loss: Divide the total perceptual loss by the number of used layers.

    Raises:
        ValueError: `vgg_type` must be one of: ``'vgg11'``, ``'vgg13'``, ``'vgg16'``, ``'vgg19'``, raise error if not.
        ValueError: `layer_weights` keys must be defined as a layer of given VGG network type.
        NotImplementedError: `loss_f` must be one of: ``'l1'``, ``'l2'``, raise error if not.
    """

    def __init__(self, layer_weights: Dict[str, float], vgg_type: str = "vgg19", criterion: str = "l1",
                 normalize_input: bool = True, normalize_loss: bool = False):
        super(PerceptualLoss, self).__init__()

        # Initialize layer names list from layer weights dictionary keys
        self.layer_names = layer_weights.keys()
        self.layer_weights = layer_weights
        # Define flags
        self.normalize_input = normalize_input
        self.normalize_loss = normalize_loss

        # Check that the VGG type is valid
        if vgg_type not in _VGG_LAYER_NAMES.keys():
            raise ValueError(f"There isn't any VGG network type with the name: {vgg_type}")

        # Initialize VGG layer names from the given VGG network type
        self._vgg_layer_names = _VGG_LAYER_NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = _insert_bn_layers(self._vgg_layer_names)

        # Check that all the given VGG layer names are defined in the network
        for layer in self.layer_names:
            if layer not in self._vgg_layer_names:
                raise ValueError(f"Layer '{layer}' is not defined in '{vgg_type}' network.")

        # Calculate the maximum layer index to avoid taking unused parameters from the VGG network
        max_layer_i = max([self._vgg_layer_names.index(name) for name in self.layer_names])

        # Retrieve a pretrained version of the VGG network
        vgg_net = getattr(vgg, vgg_type)(pretrained=True)
        # Extract the needed feature layers from the VGG network
        features = vgg_net.features[:max_layer_i + 1]
        # Create an ordered dictionary containing VGG network layers with its name
        net_layers = OrderedDict(zip(self._vgg_layer_names, features))

        # Create the modified VGG network from the feature list as a sequential module and set it to eval mode
        self.vgg_net = nn.Sequential(net_layers)
        self.vgg_net.eval()
        # Disable gradients of all parameters of this module to avoid back-propagating through the VGG network
        for param in self.parameters():
            param.requires_grad = False

        criterion = criterion.lower()
        # Define the criterion to calculate perceptual loss between feature maps
        if criterion == "l1":
            self.criterion = l1_loss
        elif criterion == "l2":
            self.criterion = mse_loss
        else:
            raise NotImplementedError(f'Perceptual loss {criterion} criterion is not supported.')

        # If we need to normalize the input image tensor, define the mean and standard deviation to be applied
        if self.normalize_input:
            # The mean and std values are for an image within the range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std',  torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # If we need to normalize the loss, divide all layer weights by the total number of layers
        if self.normalize_loss:
            w_sum = sum(self.layer_weights.values())
            self.layer_weights = {name: float(weight) / float(w_sum) for name, weight in self.layer_weights}

    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract feature maps from the input image tensor for each requested VGG layer.

        Args:
            x: Input tensor with shape (n, c, h, w).

        Returns:
            Dict of layers and its feature maps.
        """
        features: Dict[str, torch.Tensor] = {}

        # Normalize the input
        if self.normalize_input:
            x = (x - self.mean) / self.std

        # Extract network features
        for name, layer in self.vgg_net.named_children():
            # Forward-pass through the layer
            x = layer(x)
            # Store the output tensor if requested
            if name in self.layer_names:
                features[name] = x.clone()

        return features

    def forward(self, out_images: torch.Tensor, gt_images: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation method for the perceptual loss.

        Args:
            out_images: Batch of input (generated) images.
            gt_images: Batch of target (real, ground truth) images.

        Returns:
            Loss, scalar.
        """
        gen_img_features = self._extract_features(out_images)
        gt_img_features = self._extract_features(gt_images.detach())

        # Initialize the perceptual loss to 0
        perceptual_loss = 0.
        # Calculate weighted sum of distances between real and fake features
        for layer, weight in self.layer_weights.items():
            # Calculate the weighted loss between features of current layer
            layer_loss = self.criterion(gen_img_features[layer], gt_img_features[layer])
            # Append it to the total perceptual loss calculation
            perceptual_loss = perceptual_loss + (layer_loss * float(weight))

        return perceptual_loss
