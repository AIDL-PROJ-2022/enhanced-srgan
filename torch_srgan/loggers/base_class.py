import torch

from abc import ABC
from typing import Union, List


class Logger(ABC):
    def __init__(self, hr_scale_factor: int):
        self._current_step: int = 0
        self._hr_scale_factor = hr_scale_factor

    def set_current_step(self, step: int):
        self._current_step = step

    def step(self):
        self._current_step += 1

    def log_metrics(self, stage: str, dataset_target: str, metrics: dict):
        raise NotImplementedError

    def log_images(self, target: str, lr_images: torch.Tensor, out_images: torch.Tensor, gt_images: torch.Tensor):
        raise NotImplementedError

    def log_model_graph(self, model: torch.nn.Module, images: Union[torch.Tensor, List[torch.Tensor]]):
        raise NotImplementedError
