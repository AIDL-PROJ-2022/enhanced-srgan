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
        paired_img_scale (float): scale relation between image and scaled image.
        always_apply (bool): always apply transform. Default: False.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """

    def __init__(self, paired_img_scale: float, always_apply: bool = False, p: float = 1.0):
        super(PairedTransform, self).__init__(always_apply, p)
        self.paired_img_scale = paired_img_scale

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
        exp_scaled_img_size = tuple([round(v * self.paired_img_scale) for v in img_size])
        # Check that scaled image match expected dimensions
        if scaled_img_size != exp_scaled_img_size:
            raise ValueError(f"Input scaled image dimensions {scaled_img_size} aren't {self.paired_img_scale}x "
                             f"multiplication of input image {img_size}.")
        # Add extra generated parameters to original parameters
        params.update({"scaled_cols": scaled_img_size[1], "scaled_rows": scaled_img_size[0]})
        return params

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class PairedRandomCrop(PairedTransform):
    """
    Crop a random part of an input image and a scaled version of it.
    Both images should be the same with different scale.
    Output cropped scaled image will be scale times the size of the configured crop size.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        paired_img_scale (float): scale relation between image and scaled image.
        always_apply (bool): always apply transform. Default: False.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """

    def __init__(self, height: int, width: int, paired_img_scale: float, always_apply: bool = False, p: float = 1.0):
        super(PairedRandomCrop, self).__init__(paired_img_scale, always_apply, p)
        self.height = height
        self.width = width
        self._cr_size = (height, width)
        self._scaled_cr_size = (round(height * paired_img_scale), round(width * paired_img_scale))

    @staticmethod
    def _apply_rand_crop(img: np.ndarray, size: Tuple[int, int], h_start: float = 0, w_start: float = 0) -> np.ndarray:
        return cr_func.random_crop(img, *size, h_start=h_start, w_start=w_start)

    def apply(self, img: np.ndarray, h_start: float = 0, w_start: float = 0, **_params) -> np.ndarray:
        return self._apply_rand_crop(img, self._cr_size, h_start, w_start)

    def apply_scaled(self, img: np.ndarray, h_start: float = 0, w_start: float = 0, **_params) -> np.ndarray:
        return self._apply_rand_crop(img, self._scaled_cr_size, h_start, w_start)

    def get_params(self) -> Dict[str, Any]:
        return {"h_start": random.random(), "w_start": random.random()}

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return "height", "width", "paired_img_scale"


class PairedCenterCrop(PairedTransform):
    """
    Crop a center part of an input image and a scaled version of it.
    Both images should be the same with different scale.
    Output cropped scaled image will be scale times the size of the configured crop size.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        paired_img_scale (float): scale relation between image and scaled image.
        always_apply (bool): always apply transform. Default: False.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """

    def __init__(self, height: int, width: int, paired_img_scale: float, always_apply: bool = False, p: float = 1.0):
        super(PairedCenterCrop, self).__init__(paired_img_scale, always_apply, p)
        self.height = height
        self.width = width
        self._cr_size = (height, width)
        self._scaled_cr_size = (round(height * paired_img_scale), round(width * paired_img_scale))

    @staticmethod
    def _apply_center_crop(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return cr_func.center_crop(img, *size)

    def apply(self, img: np.ndarray, **_params) -> np.ndarray:
        return self._apply_center_crop(img, self._cr_size)

    def apply_scaled(self, img: np.ndarray, **_params) -> np.ndarray:
        return self._apply_center_crop(img, self._scaled_cr_size)

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return "height", "width", "paired_img_scale"


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

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return "mean", "std", "max_pixel_value"

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}
