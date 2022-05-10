"""
Base class for image pair datasets defined for test, train and validate models.
"""

__author__ = ""
__credits__ = [""]
__license__ = "GPL-3.0"
__version__ = "0.1.0"
__status__ = "Development"

from abc import ABC
from typing import Optional, Dict, List, Any

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive


class CustomImagePairDataset(Dataset, ABC):
    """
    Image pair dataset base class used by all the datasets defined to train, test and validate SR models.

    Args:
        root (string): Root directory of dataset.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    .. note::

        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """

    url = ""
    filename = ""
    tgz_md5 = ""º

    def __init__(self, scale_factor: int = 2, base_dir: str = ".data", hr_img_dir: str = "hr_img",
                 lr_img_dir: str = "lr_img", color_space: str = "RGB", needs_downscale: bool = True):
        # Define class member variables from input parameters
        self.scale_factor = scale_factor
        self.base_dir = base_dir
        self.color_space = color_space

        # Initialize image pair data dictionary ()
        self.data: List[Dict[Any, Any]] = []

    def _check_integrity(self):
        raise NotImplementedError

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.base_dir, filename=self.filename, md5=self.tgz_md5)
