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

from .base_class import ImagePairDataset


class BSDS500(ImagePairDataset):
    """
    BSDS500 image dataset class.

    Args:
        TODO
    """
    dataset_img_base_dir = "BSR/BSDS500/data/images/"

    def __init__(self, target: str, scale_factor: int = 2, patch_size: Tuple[int, int] = (96, 96),
                 base_dir: str = "data", transforms: List[A.BasicTransform] = None, download: bool = False):
        # Check that user provided a valid target
        if target not in ("train", "test", "val"):
            raise ValueError(f"Invalid BSDS500 target '{target}' provided.")
        # Define if user requested train mode or not
        train_mode = (target == "train")
        # Also, define full HR images directory
        hr_img_dir = os.path.join(BSDS500.dataset_img_base_dir, target)

        # TODO: DOWNLOAD DATASET AUTOMATICALLY

        super(BSDS500, self).__init__(scale_factor, train_mode, patch_size, base_dir, hr_img_dir, transforms)
