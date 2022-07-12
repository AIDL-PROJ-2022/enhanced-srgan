"""
Base class for model testing logging interface.
"""

__author__ = "Marc Bermejo"


import torch

from abc import ABC
from typing import Union, List


class Logger(ABC):
    """
    Base class for model testing logging interface.

    Args:
        hr_scale_factor: High-resolution image scale in respect to low resolution one.
    """

    def __init__(self, hr_scale_factor: int):
        self._current_step: int = 0
        self._hr_scale_factor = hr_scale_factor

    def set_current_step(self, step: int):
        """
        Set the current step of the logger.

        Args:
            step: New logger step.
        """
        self._current_step = step

    def step(self):
        """
        Increment logger step by 1.
        """
        self._current_step += 1

    def log_metrics(self, stage: str, dataset_target: str, metrics: dict):
        """
        Log model training metrics.

        Args:
            stage: Training stage.
            dataset_target: Target of the used dataset.
            metrics: Metrics to log.
        """
        raise NotImplementedError

    def log_images(self, target: str, lr_images: torch.Tensor, out_images: torch.Tensor, gt_images: torch.Tensor):
        """
        Log model training images.

        Args:
            target: Target of the used dataset.
            lr_images: Tensor containing the low resolution images.
            out_images: Tensor containing the generated images.
            gt_images: Tensor containing the high resolution images.
        """
        raise NotImplementedError

    def log_model_graph(self, model: torch.nn.Module, images: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Log model graph.

        Args:
            model: Model to log.
            images: Example images to perform inference.
        """
        raise NotImplementedError
