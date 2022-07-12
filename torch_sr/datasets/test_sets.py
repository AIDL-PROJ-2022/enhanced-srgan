"""
Set5 and Set14 test datasets classes implementation.
"""

__author__ = "Marc Bermejo"


import os
import albumentations as A

from typing import List

from ..utils.datasets import download_file_from_google_drive, extract_archive
from .base_class import ImagePairDataset


class _BaseTestImageSets(ImagePairDataset):
    """
    Base class for `Set5 <https://deepai.org/dataset/set5-super-resolution>` and
    `Set14 <https://deepai.org/dataset/set14-super-resolution>` datasets.

    Args:
        scale_factor: Scale factor relation between the low resolution and the high resolution images.
        base_dir: Base directory where datasets are stored.
        transforms: List of user-defined transformations to be applied to the dataset images.
        download: Flag to indicate if the dataset needs to be downloaded or not.
    """

    file_id = None
    dataset_md5sum = None
    file_name = None
    dataset_img_base_dir = None

    def __init__(self, scale_factor: int = 2, base_dir: str = "data", transforms: List[A.BasicTransform] = None,
                 download: bool = True):
        # Check that user provided a valid parameters
        if scale_factor not in (2, 3, 4):
            raise ValueError(f"Invalid DIV2K scale factor x{scale_factor} provided.")

        # Download the dataset from the given URL if requested
        if download:
            download_file_from_google_drive(self.file_id, base_dir, filename=self.file_name, md5=self.dataset_md5sum)

        print("Extracting {} to {}".format(self.file_name, base_dir))
        extract_archive(os.path.join(base_dir, self.file_name))

        # Define high and low resolution images directory
        hr_img_dir = os.path.join(self.dataset_img_base_dir, "GTmod12")
        lr_img_dir = os.path.join(self.dataset_img_base_dir, f"LRbicx{scale_factor}")

        super(_BaseTestImageSets, self).__init__(
            scale_factor, False, None, base_dir, hr_img_dir, lr_img_dir, transforms=transforms
        )


class Set5(_BaseTestImageSets):
    """
    `Set5 <https://deepai.org/dataset/set5-super-resolution>`_ Dataset.

    Args:
        scale_factor: Scale factor relation between the low resolution and the high resolution images.
        base_dir: Base directory where datasets are stored.
        transforms: List of user-defined transformations to be applied to the dataset images.
        download: Flag to indicate if the dataset needs to be downloaded or not.
    """

    file_id = "1DnHLNkcpl0wLznwAGW6CcrMMJOZY8ILz"
    dataset_md5sum = "4395d7e7e6d7ac5c9cb2a8b2acbdba5d"
    file_name = "Set5.zip"
    dataset_img_base_dir = "Set5"

    def __init__(self, scale_factor: int = 2, base_dir: str = "data", transforms: List[A.BasicTransform] = None,
                 download: bool = True):
        super(Set5, self).__init__(scale_factor, base_dir, transforms, download)


class Set14(_BaseTestImageSets):
    """
    `Set14 <https://deepai.org/dataset/set14-super-resolution>`_ Dataset.

    Args:
        scale_factor: Scale factor relation between the low resolution and the high resolution images.
        base_dir: Base directory where datasets are stored.
        transforms: List of user-defined transformations to be applied to the dataset images.
        download: Flag to indicate if the dataset needs to be downloaded or not.
    """

    file_id = "1YC6l1o8qBtkU4LUtBQbOZ5sIM-lZf7YO"
    dataset_md5sum = "efe68a0553772ace8b1c59cf8496b6f6"
    file_name = "Set14.zip"
    dataset_img_base_dir = "Set14"

    def __init__(self, scale_factor: int = 2, base_dir: str = "data", transforms: List[A.BasicTransform] = None,
                 download: bool = True):
        super(Set14, self).__init__(scale_factor, base_dir, transforms, download)
