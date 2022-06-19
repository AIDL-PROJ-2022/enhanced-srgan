from abc import ABC

from torch import Tensor


class Logger(ABC):
    def __init__(self):
        self._current_step: int = 0

    def set_current_step(self, step: int):
        self._current_step = step

    def step(self):
        self._current_step += 1

    def log_metrics(self, stage: str, dataset_target: str, metrics: dict):
        raise NotImplementedError

    def log_images(self, target: str, lr_images: Tensor, out_images: Tensor, gt_images: Tensor):
        raise NotImplementedError
