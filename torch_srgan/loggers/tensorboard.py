import os
import datetime
import torch
import wandb

from torchvision.utils import make_grid
from torch_srgan.loggers.base_class import Logger
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger(Logger):
    def __init__(self, task: str, generator: torch.nn.Module = None, discriminator: torch.nn.Module = None):
        logdir = os.path.join("logs",f"{task}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.writer = SummaryWriter(logdir)
        
    def log_stage(self, stage: str, epoch: int, train_losses: float, val_losses: float,
                  psnr_metric: float, ssim_metric: float):
        self.writer.add_scalar(f"{stage}/train_loss", train_losses, epoch)
        self.writer.add_scalar(f"{stage}/val_loss", val_losses, epoch)
        self.writer.add_scalar(f"{stage}/psnr_metric", psnr_metric, epoch)
        self.writer.add_scalar(f"{stage}/ssim_metric", ssim_metric, epoch)

    def log_image_transforms(self, epoch: int, dataset_target: str, img_transforms: dict):
        self.writer.add_images(f"{dataset_target}/img_transforms", img_transforms, epoch)

    def log_images(self, epoch: int, target: str, lr_images: torch.Tensor,
                   out_images: torch.Tensor, gt_images: torch.Tensor):
        for idx, out_image in enumerate(out_images):
            self.writer.add_image(f'{target}/image-{idx}', make_grid([out_image, gt_images[idx]]), epoch)
    
    def log_model_graph(self, model, train_loader):
        batch, _ = next(iter(train_loader))
        # TODO: This could be an error and have to be checked and deleted this line
        batch = torch.device(batch.cuda() if torch.cuda.is_available() else batch)
        self.writer.add_graph(model, batch)