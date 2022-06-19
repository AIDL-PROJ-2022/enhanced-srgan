"""
DIV2K dataset class implementation.
"""

__author__ = "Marc Bermejo"
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

import albumentations as A

from typing import List, Tuple

from ..utils.datasets import download_and_extract_archive
from .base_class import ImagePairDataset


class DIV2K(ImagePairDataset):
    """`DIV2K <https://data.vision.ee.ethz.ch/cvl/DIV2K>`_ Dataset.

    Args:
        TODO
    """

    base_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/"
    resources = {
        "DIV2K_train_LR_bicubic_X2.zip": "9a637d2ef4db0d0a81182be37fb00692",
        "DIV2K_train_LR_unknown_X2.zip": "1396d023072c9aaeb999c28b81315233",
        "DIV2K_valid_LR_bicubic_X2.zip": "1512c9a3f7bde2a1a21a73044e46b9cb",
        "DIV2K_valid_LR_unknown_X2.zip": "d319bd9033573d21de5395e6454f34f8",
        "DIV2K_train_LR_bicubic_X3.zip": "ad80b9fe40c049a07a8a6c51bfab3b6d",
        "DIV2K_train_LR_unknown_X3.zip": "4e651308aaa54d917fb1264395b7f6fa",
        "DIV2K_valid_LR_bicubic_X3.zip": "18b1d310f9f88c13618c287927b29898",
        "DIV2K_valid_LR_unknown_X3.zip": "05184168e3608b5c539fbfb46bcade4f",
        "DIV2K_train_LR_bicubic_X4.zip": "76c43ec4155851901ebbe8339846d93d",
        "DIV2K_train_LR_unknown_X4.zip": "e3c7febb1b3f78bd30f9ba15fe8e3956",
        "DIV2K_valid_LR_bicubic_X4.zip": "21962de700c8d368c6ff83314480eff0",
        "DIV2K_valid_LR_unknown_X4.zip": "8ac3413102bb3d0adc67012efb8a6c94",
        "DIV2K_train_LR_x8.zip": "613db1b855721b3d2b26f4194a1d22a6",
        "DIV2K_train_LR_mild.zip": "807b3e3a5156f35bd3a86c5bbfb674bc",
        "DIV2K_train_LR_difficult.zip": "5a8f2b9e0c5f5ed0dac271c1293662f4",
        "DIV2K_train_LR_wild.zip": "d00982366bffee7c4739ba7ff1316b3b",
        "DIV2K_valid_LR_x8.zip": "c5aeea2004e297e9ff3abfbe143576a5",
        "DIV2K_valid_LR_mild.zip": "8c433f812ca532eed62c11ec0de08370",
        "DIV2K_valid_LR_difficult.zip": "1620af11bf82996bc94df655cb6490fe",
        "DIV2K_valid_LR_wild.zip": "aacae8db6bec39151ca5bb9c80bf2f6c",
        "DIV2K_train_HR.zip": "bdc2d9338d4e574fe81bf7d158758658",
        "DIV2K_valid_HR.zip": "9fcdda83005c5e5997799b69f955ff88",
    }

    def __init__(self, target: str, scale_factor: int = 2, patch_size: Tuple[int, int] = (128, 128),
                 bicubic_downscale: bool = True, base_dir: str = "data", transforms: List[A.BasicTransform] = None,
                 download: bool = True):

        # Check that user provided a valid parameters
        if target not in ("train", "test", "val"):
            raise ValueError(f"Invalid DIV2K target '{target}' provided.")
        if scale_factor not in (2, 3, 4, 8):
            raise ValueError(f"Invalid DIV2K scale factor x{scale_factor} provided.")

        # Define internal constants
        train_mode = (target == "train")
        target_name = 'train' if train_mode else 'valid'
        interpolation = 'bicubic' if bicubic_downscale else 'unknown'
        filename_hr = f"DIV2K_{target_name}_HR.zip"
        if scale_factor == 8:
            filename_lr = f"DIV2K_{target_name}_LR_X{scale_factor}.zip"
        else:
            filename_lr = f"DIV2K_{target_name}_LR_{interpolation}_X{scale_factor}.zip"

        # Download the dataset from the given URL if requested
        if download:
            download_and_extract_archive(f"{self.base_url}{filename_hr}", base_dir, md5=self.resources[filename_hr])
            download_and_extract_archive(f"{self.base_url}{filename_lr}", base_dir, md5=self.resources[filename_lr])

        # Define high and low resolution images directory
        hr_img_dir = f"DIV2K_{target_name}_HR"
        if scale_factor == 8:
            lr_img_dir = f"DIV2K_{target_name}_LR/X{scale_factor}"
        else:
            lr_img_dir = f"DIV2K_{target_name}_LR_{interpolation}/X{scale_factor}"

        super(DIV2K, self).__init__(
            scale_factor, train_mode, patch_size, base_dir, hr_img_dir, lr_img_dir, transforms=transforms
        )
