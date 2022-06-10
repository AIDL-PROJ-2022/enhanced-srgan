"""
This module implement all custom transforms needed for the image augmentation pipeline.
"""

__author__ = "Marc Bermejo"
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

import random
import numpy as np

from typing import Dict, Callable, Any, Tuple
from albumentations import BasicTransform, ImageOnlyTransform
from albumentations.augmentations.crops import functional as cr_func


class PairedTransform(BasicTransform):
    """
    Base transform for image superresolution tasks.
    Both input images should be the same with different scale.

    Args:
        paired_img_scale (int): scale relation between image and scaled image. Must be an integer >= 1.
        always_apply (bool): always apply transform. Default: False.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """

    def __init__(self, paired_img_scale: int, always_apply: bool = False, p: float = 1.0):
        super(PairedTransform, self).__init__(always_apply, p)
        self.paired_img_scale = int(paired_img_scale)
        if self.paired_img_scale < 1:
            raise ValueError("Paired image scale must be an integer equal or bigger than 1.")

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "scaled_image": self.apply_scaled
        }

    def apply_scaled(self, img: np.ndarray, h_start: float = 0, w_start: float = 0, **_params) -> np.ndarray:
        raise NotImplementedError

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Call base class method
        params = super().update_params(params, **kwargs)
        # Retrieve both images dimensions
        img_size = kwargs["image"].shape[:2]
        scaled_img_size = kwargs["scaled_image"].shape[:2]
        exp_scaled_img_size = tuple([int(dim * self.paired_img_scale) for dim in img_size])
        # Check that scaled image match expected dimensions
        if scaled_img_size != exp_scaled_img_size:
            raise ValueError(f"Input scaled image dimensions {scaled_img_size} aren't {self.paired_img_scale}x "
                             f"multiplication of input image {img_size}.")
        # Add extra generated parameters to original parameters
        params.update({"scaled_cols": scaled_img_size[1], "scaled_rows": scaled_img_size[0]})
        return params

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class PairedCrop(PairedTransform):
    """
    Base transform for paired image cropping tasks.
    Both images should be the same with different scale.
    Output cropped scaled image will be scale times the size of the configured crop size.

    Args:
        scaled_cr_size (tuple): Size of the scaled crop. Must be a (height, width) tuple.
        paired_img_scale (int): scale relation between image and scaled image. Must be an integer >= 1.
        always_apply (bool): always apply transform. Default: False.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """
    def __init__(self, scaled_cr_size: Tuple[int, int], paired_img_scale: int,
                 always_apply: bool = False, p: float = 1.0):
        super(PairedCrop, self).__init__(paired_img_scale, always_apply, p)
        self._check_and_set_scaled_cr_size(scaled_cr_size)

    def _check_and_set_scaled_cr_size(self, scaled_cr_size: Tuple[int, int]):
        # Check that scaled crop size is multiple of given scale
        if scaled_cr_size[0] % self.paired_img_scale != 0 or scaled_cr_size[1] % self.paired_img_scale != 0:
            raise ValueError(
                f"Given scaled image crop size {scaled_cr_size} isn't a multiple of "
                f"paired image scale {self.paired_img_scale}."
            )
        # Set scaled crop size and low resolution image crop size
        self.scaled_cr_size = scaled_cr_size
        self.cr_size = tuple(dim // self.paired_img_scale for dim in self.scaled_cr_size)

    def _get_scaled_crop_coords(self, x_min: int, y_min: int, crop_h: int, crop_w: int) -> Tuple[int, int, int, int]:
        # If scaled crop was requested, calculate its coordinates ensuring that these will be multiple of the scale to
        # ensure equivalent position between the two crops.
        y_min -= y_min % self.paired_img_scale
        y_max = y_min + crop_h
        x_min -= x_min % self.paired_img_scale
        x_max = x_min + crop_w
        # Return calculated values
        return x_min, y_min, x_max, y_max

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return "cr_size", "scaled_cr_size", "paired_img_scale"


class PairedRandomCrop(PairedCrop):
    """
    Crop a random part of an input image and a scaled version of it.
    Both images should be the same with different scale.
    Output cropped scaled image will be scale times the size of the configured crop size.

    Args:
        scaled_cr_size (tuple): Size of the scaled crop. Must be a (height, width) tuple.
        paired_img_scale (int): scale relation between image and scaled image. Must be an integer >= 1.
        always_apply (bool): always apply transform. Default: False.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """
    def __init__(self, scaled_cr_size: Tuple[int, int], paired_img_scale: int,
                 always_apply: bool = False, p: float = 1.0):
        super(PairedRandomCrop, self).__init__(scaled_cr_size, paired_img_scale, always_apply, p)

    def apply(self, img: np.ndarray, h_start: float = 0, w_start: float = 0, **_params) -> np.ndarray:
        # Get scaled crop height and width from its size
        crop_h, crop_w = self.cr_size
        # Return cropped image
        return cr_func.random_crop(img, crop_h, crop_w, h_start, w_start)

    def apply_scaled(self, img: np.ndarray, h_start: float = 0, w_start: float = 0, **_params) -> np.ndarray:
        # Get image height and width
        img_h, img_w = img.shape[:2]
        # Get scaled crop height and width from its size
        crop_h, crop_w = self.scaled_cr_size
        # Calculate X and Y start position
        y_min = int((img_h - crop_h) * h_start)
        x_min = int((img_w - crop_w) * w_start)
        # Calculate crop coordinates
        crop_coords = self._get_scaled_crop_coords(x_min, y_min, crop_h, crop_w)
        # Return cropped image
        return cr_func.crop(img, *crop_coords)

    def get_params(self) -> Dict[str, Any]:
        return {"h_start": random.random(), "w_start": random.random()}


class PairedCenterCrop(PairedCrop):
    """
    Crop a center part of an input image and a scaled version of it.
    Both images should be the same with different scale.
    Output cropped scaled image will be scale times the size of the configured crop size.

    Args:
        scaled_cr_size (tuple): Size of the scaled crop. Must be a (height, width) tuple.
        paired_img_scale (int): scale relation between image and scaled image. Must be an integer >= 1.
        always_apply (bool): always apply transform. Default: False.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """
    def __init__(self, scaled_cr_size: Tuple[int, int], paired_img_scale: int,
                 always_apply: bool = False, p: float = 1.0):
        super(PairedCenterCrop, self).__init__(scaled_cr_size, paired_img_scale, always_apply, p)

    def apply(self, img: np.ndarray, **_params) -> np.ndarray:
        # Get scaled crop height and width from its size
        crop_h, crop_w = self.cr_size
        # Return cropped image
        return cr_func.center_crop(img, crop_h, crop_w)

    def apply_scaled(self, img: np.ndarray, **_params) -> np.ndarray:
        # Get image height and width
        img_h, img_w = img.shape[:2]
        # Get scaled crop height and width from its size
        crop_h, crop_w = self.scaled_cr_size
        # Calculate X and Y start position
        y_min = (img_h - crop_h) // 2
        x_min = (img_w - crop_w) // 2
        # Calculate crop coordinates
        crop_coords = self._get_scaled_crop_coords(x_min, y_min, crop_h, crop_w)
        # Return cropped image
        return cr_func.crop(img, *crop_coords)


class SimpleNormalize(ImageOnlyTransform):
    """
    Simple normalization by applying the formula: `img = img / max_pixel_value`

    Args:
        max_pixel_value (float): maximum possible pixel value.
        always_apply (bool): always apply transform. Default: False.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, max_pixel_value: float = 255.0, always_apply: bool = False, p: float = 1.0):
        super(SimpleNormalize, self).__init__(always_apply, p)
        self.max_pixel_value = float(max_pixel_value)

    def apply(self, img: np.ndarray, **_params) -> np.ndarray:
        return img / self.max_pixel_value

    def get_transform_init_args_names(self) -> Tuple[str]:
        return "max_pixel_value",

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}
