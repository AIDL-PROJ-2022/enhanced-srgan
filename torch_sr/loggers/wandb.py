"""
Wandb model testing logging interface.
"""

__author__ = "Raul Puente, Marc Bermejo"


import os
import datetime
import torch
import wandb
import git

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from typing import Union, List

from .base_class import Logger


class WandbLogger(Logger):
    """
    Wandb model testing logging interface class.

    Args:
        proj_name: Project name.
        entity_name: Wandb entity name.
        task: Task name.
        hr_scale_factor: High-resolution image scale in respect to low resolution one.
        generator: Generator PyTorch model.
        discriminator: Discriminator PyTorch model.
        config: Training hyper-parameters.
    """

    def __init__(self, proj_name: str, entity_name: str, task: str, hr_scale_factor: int,
                 generator: torch.nn.Module = None, discriminator: torch.nn.Module = None, config: dict = None):
        super(WandbLogger, self).__init__(hr_scale_factor)

        logdir = os.path.join("logs", f"{task}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        self._init_wandb(
            logdir=logdir, proj_name=proj_name, entity_name=entity_name,
            task=task, config=config, generator=generator, discriminator=discriminator
        )
        self._init_tensorboard(logdir=logdir)

    @staticmethod
    def _init_wandb(logdir: str, proj_name: str, entity_name: str, task: str, generator: torch.nn.Module = None,
                    discriminator: torch.nn.Module = None, config: dict = None):
        """
        Initialize Wandb logging.

        Args:
            logdir: Directory where logging data will be stored.
            entity_name: Wandb entity name.
            task: Task name.
            generator: Generator PyTorch model.
            discriminator: Discriminator PyTorch model.
            config: Training hyper-parameters.
        """
        if config is None:
            config = dict()

        # Workaround to fix progress bars: disable wandb console
        os.environ["WANDB_CONSOLE"] = "off"

        # Initialize wandb logger
        wandb.login()
        repo = git.Repo(search_parent_directories=True)
        config['git_sha'] = repo.head.object.hexsha
        wandb.tensorboard.patch(root_logdir=logdir)
        wandb.init(project=proj_name, entity=entity_name,
                   config=config, save_code=True)
        wandb.run.name = f'{task}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

        # Log generator model
        if generator is not None:
            wandb.watch(generator, log="all")
        # Log discriminator model
        if discriminator is not None:
            wandb.watch(discriminator, log="all")

    def _init_tensorboard(self, logdir: str):
        """
        Create tensordboard summary writer object.

        Args:
            logdir: Directory where logging data will be stored.
        """
        self.writer = SummaryWriter(logdir)

    def log_metrics(self, stage: str, dataset_target: str, metrics: dict):
        """
        Log model training metrics.

        Args:
            stage: Training stage.
            dataset_target: Target of the used dataset.
            metrics: Metrics to log.
        """
        for name, value in metrics.items():
            self.writer.add_scalar(f"{dataset_target}/{stage}/{name}", value, self._current_step)

    def log_images(self, target: str, lr_images: torch.Tensor, out_images: torch.Tensor, gt_images: torch.Tensor):
        """
        Log model training images.

        Args:
            target: Target of the used dataset.
            lr_images: Tensor containing the low resolution images.
            out_images: Tensor containing the generated images.
            gt_images: Tensor containing the high resolution images.
        """
        # Upscale LR images
        lr_images = F.interpolate(lr_images, scale_factor=self._hr_scale_factor, mode='nearest-exact')
        # Define images by creating a grid between out and ground truth ones
        images = [make_grid([lr_image, out_images[idx], gt_images[idx]]) for idx, lr_image in enumerate(lr_images)]
        # Write them to wandb
        wandb.log({
            f'{target}/images': [wandb.Image(im, caption="LR / GEN / GT") for im in images]
        })

    def log_model_graph(self, model: torch.nn.Module, images: Union[torch.Tensor, List[torch.Tensor]]):
        """
        Log model graph.

        Args:
            model: Model to log.
            images: Example images to perform inference.
        """
        # Graph given model
        self.writer.add_graph(model, images)
