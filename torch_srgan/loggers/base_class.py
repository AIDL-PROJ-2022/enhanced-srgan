from abc import ABC

from torch import Tensor


class Logger(ABC):
    def log_stage(self, stage: str, epoch: int, train_losses: float, val_losses: float,
                  psnr_metric: float, ssim_metric: float):
        raise NotImplementedError

    def log_image_transforms(self, epoch: int, dataset_target: str, img_transforms: dict):
        raise NotImplementedError

    def log_images(self, epoch: int, target: str, lr_images: Tensor, out_images: Tensor, gt_images: Tensor):
        raise NotImplementedError
