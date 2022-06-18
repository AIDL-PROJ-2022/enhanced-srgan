import os
import datetime
import torch
import wandb

from torchvision.utils import make_grid
from torch_srgan.loggers.base_class import Logger


class WandbLogger(Logger):
    def __init__(self, proj_name: str, entity_name: str, task: str,
                 generator: torch.nn.Module = None, discriminator: torch.nn.Module = None):
        # Workaround to fix progress bars: disable wandb console
        os.environ["WANDB_CONSOLE"] = "off"
        # os.environ['WANDB_IGNORE_GLOBS'] = '*.pyc'
        os.environ['WANDB_IGNORE_GLOBS'] = 'loggers/*'
        # Initialize wandb logger
        wandb.login()
        wandb.init(project=proj_name, entity=entity_name)
        wandb.run.name = f'{task}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        # TODO: Commented because space and not relevant information on wandb
        # artifact = wandb.Artifact('ESRGAN', type='dataset')
        # artifact.add_dir('torch_srgan/')
        # wandb.log_artifact(artifact)
        # Log generator model
        if generator is not None:
            wandb.watch(generator, log="all")
        # Log discriminator model
        if discriminator is not None:
            wandb.watch(discriminator, log="all")

    def log_stage(self, stage: str, epoch: int, train_losses: float, val_losses: float,
                  psnr_metric: float, ssim_metric: float):
        wandb.log({f"{stage}/train_loss": train_losses}, step=epoch)
        wandb.log({f"{stage}/val_loss": val_losses}, step=epoch)
        wandb.log({f"{stage}/psnr_metric": psnr_metric}, step=epoch)
        wandb.log({f"{stage}/ssim_metric": ssim_metric}, step=epoch)

    def log_image_transforms(self, epoch: int, dataset_target: str, img_transforms: dict):
        wandb.log({f"{dataset_target}/img_transforms": img_transforms}, step=epoch)

    def log_images(self, epoch: int, target: str, lr_images: torch.Tensor,
                   out_images: torch.Tensor, gt_images: torch.Tensor):
        images = []
        for idx, out_image in enumerate(out_images):
            images.append(make_grid([out_image, gt_images[idx]]))

        wandb.log({f'{target}/images': [wandb.Image(im) for im in images]}, step=epoch)

    def log_model_graph(self, model, train_loader):
        # Not possible for wandb
        pass