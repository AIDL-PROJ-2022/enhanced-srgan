import functools
from typing import Callable

from torch import nn

Conv2d: Callable[..., nn.Module] = functools.partial(
    nn.Conv2d, kernel_size=(3, 3), padding=1
)
LeakyReLU: Callable[..., nn.Module] = functools.partial(
    nn.LeakyReLU, negative_slope=0.2, inplace=True
)
