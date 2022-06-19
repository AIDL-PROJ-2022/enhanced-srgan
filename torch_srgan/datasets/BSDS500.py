"""
BSDS500 dataset class implementation.
"""

__author__ = "Marc Bermejo"
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

import os
import albumentations as A

from typing import List, Tuple

from ..utils.datasets import download_and_extract_archive
from .base_class import ImagePairDataset


class BSDS500(ImagePairDataset):
    """
    BSDS500 image dataset class.

    Args:
        TODO
    """
    dataset_url = "https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    dataset_md5sum = "0b17383ff0b21fe43d2f28e624203169"

    dataset_img_base_dir = "BSR/BSDS500/data/images/"

    def __init__(self, target: str, scale_factor: int = 2, patch_size: Tuple[int, int] = (128, 128),
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
