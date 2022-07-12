"""
Modules needed by super-resolution models.
"""

__author__ = "Marc Bermejo"


from .misc import Conv2dK3, LeakyReLUSlopeDot2
from .rrdb import ResidualInResidualDenseBlock
from .subpixel_conv import SubPixelConv
from .interp_upscale import InterpUpscale
