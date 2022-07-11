"""
This module implement all custom transforms needed for the image augmentation pipeline.
"""

__author__ = "Marc Bermejo"


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
        paired_img_scale: scale relation between image and scaled image. Must be an integer bigger than 1.
        scaled_img_target_key: expected key of the scaled image during transform execution.
        always_apply: flag to indicate if transform will be applied always.
        p: probability of applying the transform.

    Raises:
        ValueError: `paired_img_scale` must be bigger than 1.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """

    def __init__(self, paired_img_scale: int, scaled_img_target_key: str = "scaled_image",
                 always_apply: bool = False, p: float = 1.0):
        super(PairedTransform, self).__init__(always_apply, p)

        if paired_img_scale < 1:
            raise ValueError("Paired image scale must be an integer equal or bigger than 1.")

        self.paired_img_scale = int(paired_img_scale)
        self.scaled_img_target_key = scaled_img_target_key

    @property
    def targets(self) -> Dict[str, Callable]:
        """
        Define image transformation targets.

        Returns:
            A dict containing the target name as the key and its transformation callback.
        """
        return {
            "image": self.apply,
            self.scaled_img_target_key: self.apply_scaled
        }

    def apply(self, img: np.ndarray, **_params) -> np.ndarray:
        """
        Apply transformation to image.

        Args:
            img: Image data array. Expected to be of dimension (H, W, CH).
            **_params: Extra transformation parameters. Not used.

        Returns:
            Transformed image.
        """
        raise NotImplementedError

    def apply_scaled(self, img: np.ndarray, **_params) -> np.ndarray:
        """
        Apply transformation to scaled image.

        Args:
            img: Scaled image data array. Expected to be of dimension (H, W, CH).
            **_params: Extra transformation parameters. Not used.

        Returns:
            Transformed scaled image.
        """
        raise NotImplementedError

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Update parameters depending on the input image.

        Args:
            params: Image transformation parameters.
            **kwargs: Image transformation targets.

        Returns:
            Updated image transformation parameters.
        """
        # Call base class method
        params = super().update_params(params, **kwargs)
        # Retrieve both images dimensions
        img_size = kwargs["image"].shape[:2]
        scaled_img_size = kwargs[self.scaled_img_target_key].shape[:2]
        exp_scaled_img_size = tuple([int(dim * self.paired_img_scale) for dim in img_size])
        # Check that scaled image match expected dimensions
        if scaled_img_size != exp_scaled_img_size:
            raise ValueError(f"Input scaled image dimensions {scaled_img_size} aren't {self.paired_img_scale}x "
                             f"multiplication of input image {img_size}.")
        # Add extra generated parameters to original parameters
        params.update({"scaled_cols": scaled_img_size[1], "scaled_rows": scaled_img_size[0]})
        return params

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get parameter values that depend on targets.

        Args:
            params: Transformation parameters.

        Returns:
            Empty dictionary.
        """
        return dict()

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """
        Get transformation constructor parameter names.

        Returns:
            A tuple containing transformation parameter names.
        """
        return "paired_img_scale",


class PairedCrop(PairedTransform):
    """
    Base transform for paired image cropping tasks.
    Both images should be the same with different scale.
    Output cropped scaled image will be scale times the size of the configured crop size.

    Args:
        scaled_cr_size: Size of the scaled crop. Must be a (height, width) tuple.
        paired_img_scale: scale relation between image and scaled image. Must be an integer >= 1.
        scaled_img_target_key: expected key of the scaled image during transform execution.
        always_apply: flag to indicate if transform will be applied always.
        p: probability of applying the transform.

    Raises:
        ValueError: Scaled crop size must be a multiple of the given scale.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """

    def __init__(self, scaled_cr_size: Tuple[int, int], paired_img_scale: int,
                 scaled_img_target_key: str = "scaled_image", always_apply: bool = True, p: float = 1.0):
        super(PairedCrop, self).__init__(paired_img_scale, scaled_img_target_key, always_apply, p)

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
        """
        Retrieve scaled crop coordinates from given X and Y start positions and the crop dimensions.
        This function will ensure that the crop coordinates will be multiple of the image scale to have
        equivalent position between the paired crops.

        Args:
            x_min: Scaled crop start X coordinate.
            y_min: Scaled crop start Y coordinate.
            crop_h: Scaled crop height.
            crop_w: Scaled crop width.

        Returns:
            A tuple containing the full crop vertex in the form (x_min, y_min, x_max, y_max).
        """
        # If scaled crop was requested, calculate its coordinates ensuring that these will be multiple of the scale to
        # ensure equivalent position between the two crops.
        y_min -= y_min % self.paired_img_scale
        y_max = y_min + crop_h
        x_min -= x_min % self.paired_img_scale
        x_max = x_min + crop_w
        # Return calculated values
        return x_min, y_min, x_max, y_max

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        """
        Get transformation constructor parameter names.

        Returns:
            A tuple containing transformation parameter names.
        """
        return "cr_size", "scaled_cr_size", "paired_img_scale"

    def apply(self, img: np.ndarray, **_params) -> np.ndarray:
        """
        Apply transformation to image.

        Args:
            img: Image data array. Expected to be of dimension (H, W, CH).
            **_params: Extra transformation parameters. Not used.

        Returns:
            Transformed image.
        """
        raise NotImplementedError

    def apply_scaled(self, img: np.ndarray, **_params) -> np.ndarray:
        """
        Apply transformation to scaled image.

        Args:
            img: Scaled image data array. Expected to be of dimension (H, W, CH).
            **_params: Extra transformation parameters. Not used.

        Returns:
            Transformed scaled image.
        """
        raise NotImplementedError


class PairedRandomCrop(PairedCrop):
    """
    Crop a random part of an input image and a scaled version of it.
    Both images should be the same with different scale.
    Output cropped scaled image will be scale times the size of the configured crop size.

    Args:
        scaled_cr_size: Size of the scaled crop. Must be a (height, width) tuple.
        paired_img_scale: scale relation between image and scaled image. Must be an integer >= 1.
        scaled_img_target_key: expected key of the scaled image during transform execution.
        always_apply: flag to indicate if transform will be applied always.
        p: probability of applying the transform.

    Raises:
        ValueError: Scaled crop size must be a multiple of the given scale.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """

    def __init__(self, scaled_cr_size: Tuple[int, int], paired_img_scale: int,
                 scaled_img_target_key: str = "scaled_image", always_apply: bool = True, p: float = 1.0):
        super(PairedRandomCrop, self).__init__(scaled_cr_size, paired_img_scale, scaled_img_target_key, always_apply, p)

    def apply(self, img: np.ndarray, h_start: float = 0, w_start: float = 0, **_params) -> np.ndarray:
        """
        Apply a random crop to the image.

        Args:
            img: Image data array. Expected to be of dimension (H, W, CH).
            h_start: Crop start position in the vertical axis.
            w_start: Crop start position in the horizontal axis.
            **_params: Extra transformation parameters. Not used.

        Returns:
            Transformed image.
        """
        # Get image height and width
        img_h, img_w = img.shape[:2]
        # Get crop height and width from its size
        crop_h, crop_w = self.cr_size
        # Calculate X and Y start position
        y_min = int((img_h - crop_h) * h_start)
        x_min = int((img_w - crop_w) * w_start)
        # Calculate X and Y end position
        y_max = y_min + crop_h
        x_max = x_min + crop_w
        # Return cropped image
        return cr_func.crop(img, x_min, y_min, x_max, y_max)

    def apply_scaled(self, img: np.ndarray, h_start: float = 0, w_start: float = 0, **_params) -> np.ndarray:
        """
        Apply a random crop to the scaled image.

        Args:
            img: Scaled image data array. Expected to be of dimension (H, W, CH).
            h_start: Crop start position in the vertical axis.
            w_start: Crop start position in the horizontal axis.
            **_params: Extra transformation parameters. Not used.

        Returns:
            Transformed image.
        """
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
        """
        Get the parameters used to perform the image transformation.

        Returns:
            The parameters used to perform the image transformation.
        """
        return {"h_start": random.random(), "w_start": random.random()}


class PairedCenterCrop(PairedCrop):
    """
    Crop a center part of an input image and a scaled version of it.
    Both images should be the same with different scale.
    Output cropped scaled image will be scale times the size of the configured crop size.

    Args:
        scaled_cr_size: Size of the scaled crop. Must be a (height, width) tuple.
        paired_img_scale: scale relation between image and scaled image. Must be an integer >= 1.
        scaled_img_target_key: expected key of the scaled image during transform execution.
        always_apply: flag to indicate if transform will be applied always.
        p: probability of applying the transform.

    Raises:
        ValueError: Scaled crop size must be a multiple of the given scale.

    Targets:
        image, scaled_image

    Image types:
        uint8, float32
    """

    def __init__(self, scaled_cr_size: Tuple[int, int], paired_img_scale: int,
                 scaled_img_target_key: str = "scaled_image", always_apply: bool = True, p: float = 1.0):
        super(PairedCenterCrop, self).__init__(scaled_cr_size, paired_img_scale, scaled_img_target_key, always_apply, p)

    def apply(self, img: np.ndarray, **_params) -> np.ndarray:
        """
        Apply a center crop to the image.

        Args:
            img: Image data array. Expected to be of dimension (H, W, CH).
            **_params: Extra transformation parameters. Not used.

        Returns:
            Transformed image.
        """
        # Get scaled crop height and width from its size
        crop_h, crop_w = self.cr_size
        # Return cropped image
        return cr_func.center_crop(img, crop_h, crop_w)

    def apply_scaled(self, img: np.ndarray, **_params) -> np.ndarray:
        """
        Apply a center crop to the scaled image.

        Args:
            img: Scaled image data array. Expected to be of dimension (H, W, CH).
            **_params: Extra transformation parameters. Not used.

        Returns:
            Transformed image.
        """
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
    Simple normalization by applying the formula: :math:`img = \frac{img}{maxPixelValue}`

    Args:
        max_pixel_value: maximum possible pixel value.
        always_apply: flag to indicate if transform will be applied always.
        p: probability of applying the transform.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, max_pixel_value: float = 255.0, always_apply: bool = True, p: float = 1.0):
        super(SimpleNormalize, self).__init__(always_apply, p)
        self.max_pixel_value = float(max_pixel_value)

    def apply(self, img: np.ndarray, **_params) -> np.ndarray:
        """
        Apply simple normalization to image.

        Args:
            img: Image data array. Expected to be of dimension (H, W, CH).
            **_params: Extra transformation parameters. Not used.

        Returns:
            Transformed image.
        """
        return img.astype(np.float32) / float(self.max_pixel_value)

    def get_transform_init_args_names(self) -> Tuple[str]:
        """
        Get transformation constructor parameter names.

        Returns:
            A tuple containing transformation parameter names.
        """
        return "max_pixel_value",

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get parameter values that depend on targets.

        Args:
            params: Transformation parameters.

        Returns:
            Empty dictionary.
        """
        return dict()
