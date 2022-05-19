"""
Base class for image pair datasets defined for test, train and validate models.
"""

__author__ = "Marc Bermejo"
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

import random
import numpy as np

from typing import Dict, Callable, Any, Tuple
from albumentations import BasicTransform
from albumentations.augmentations.crops import functional as cr_func


class PairedRandomCrop(BasicTransform):
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

    def __init__(self, height: int, width: int, paired_img_scale: float, always_apply: bool = False, p: int = 1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.paired_img_scale = paired_img_scale
        self._cr_size = (height, width)
        self._scaled_cr_size = (round(height * paired_img_scale), round(width * paired_img_scale))

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "scaled_image": self.apply_scaled
        }

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

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
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
