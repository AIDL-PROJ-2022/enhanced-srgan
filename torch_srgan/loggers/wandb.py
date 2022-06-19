import os
import datetime
import wandb

from torch import Tensor
from torch.nn import Module

from .base_class import Logger


class WandbLogger(Logger):
    def __init__(self, proj_name: str, entity_name: str, task: str,
                 generator: Module = None, discriminator: Module = None):
        super(WandbLogger, self).__init__()
        # Workaround to fix progress bars: disable wandb console
        os.environ["WANDB_CONSOLE"] = "off"
        # Initialize wandb logger
        wandb.login()
        wandb.init(project=proj_name, entity=entity_name)
        wandb.run.name = f'{task}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        # Log generator model
        if generator is not None:
            wandb.watch(generator, log="all")
        # Log discriminator model
        if discriminator is not None:
            wandb.watch(discriminator, log="all")

    def log_metrics(self, stage: str, dataset_target: str, metrics: dict):
        for name, value in metrics.items():
            wandb.log({f"{dataset_target}/{stage}/{name}": value}, step=self._current_step)

    def log_images(self, target: str, lr_images: Tensor, out_images: Tensor, gt_images: Tensor):
        wandb.log({f'{target}/lr_images': [wandb.Image(im) for im in lr_images]}, step=self._current_step)
        wandb.log({f'{target}/out_images': [wandb.Image(im) for im in out_images]}, step=self._current_step)
        wandb.log({f'{target}/gt_images': [wandb.Image(im) for im in gt_images]}, step=self._current_step)
