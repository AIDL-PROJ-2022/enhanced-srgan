"""
BSDS500 dataset class implementation.
"""

__author__ = "Marc Bermejo"


import os
import albumentations as A

from typing import List, Tuple, Optional

from ..utils.datasets import download_and_extract_archive
from .base_class import ImagePairDataset


class BSDS500(ImagePairDataset):
    """
    BSDS500 image dataset class.

    Args:
        target: Dataset split target. Allowed values are: ``'train'``, ``'test'``, ``'val'``
        scale_factor: Scale factor relation between the low resolution and the high resolution images.
        patch_size: High-resolution image crop size. Needs to be a tuple of (H, W).
            If set to None, any crop transform will be applied.
        base_dir: Base directory where datasets are stored.
        transforms: List of user-defined transformations to be applied to the dataset images.
        download: Flag to indicate if the dataset needs to be downloaded or not.

    Raises:
        ValueError: `target` must be one of: ``'train'``, ``'test'``, ``'val'``, raise error if not.
    """

    dataset_url = "https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    dataset_md5sum = "0b17383ff0b21fe43d2f28e624203169"

    dataset_img_base_dir = "BSR/BSDS500/data/images/"

    def __init__(self, target: str, scale_factor: int = 2, patch_size: Optional[Tuple[int, int]] = (128, 128),
                 base_dir: str = "data", transforms: List[A.BasicTransform] = None, download: bool = True):
        # Check that user provided a valid target
        if target not in ("train", "test", "val"):
            raise ValueError(f"Invalid BSDS500 target '{target}' provided.")
        # Define if user requested train mode or not
        train_mode = (target == "train")
        # Also, define full HR images directory
        hr_img_dir = os.path.join(BSDS500.dataset_img_base_dir, target)

        # Download the dataset from the given URL if requested
        if download:
            download_and_extract_archive(BSDS500.dataset_url, base_dir, md5=BSDS500.dataset_md5sum)

        super(BSDS500, self).__init__(
            scale_factor, train_mode, patch_size, base_dir, hr_img_dir, transforms=transforms
        )
