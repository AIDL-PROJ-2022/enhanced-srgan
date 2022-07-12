"""
Miscellaneous modules.
"""

__author__ = "Marc Bermejo"


import functools
from typing import Callable

from torch import nn


Conv2dK3: Callable[..., nn.Module] = functools.partial(
    nn.Conv2d, kernel_size=(3, 3), padding=1
)

LeakyReLUSlopeDot2: Callable[..., nn.Module] = functools.partial(
    nn.LeakyReLU, negative_slope=0.2, inplace=True
)
