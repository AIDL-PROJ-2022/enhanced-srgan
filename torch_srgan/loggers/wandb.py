from torch import Tensor
import wandb
from torch_srgan.loggers.base_class import Logger
import datetime
import numpy as np

class WandbLogger(Logger):
    def __init__(self ,name_wandb_project, name_wandb_entity, task, model) -> None:
        wandb.login()
        wandb.init(project=name_wandb_project, entity=name_wandb_entity)
        wandb.run.name = f'{task}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        wandb.watch(
            model, log="all"
        )

    def log_stage(self, stage, epoch, train_losses, val_losses, psnr_metric, ssim_metric):
        wandb.log({f"{stage}/train_loss_avg": train_losses.avg , "epoch": epoch})
        wandb.log({f"{stage}/traing_loss_val": train_losses.val , "epoch": epoch})
        # wandb.log({f"{stage}/traing_loss_sum": train_losses.sum , "epoch": epoch})
        # wandb.log({f"{stage}/traing_loss_count": train_losses.count , "epoch": epoch})

        wandb.log({f"{stage}/val_loss_avg": val_losses.avg , "epoch": epoch})
        wandb.log({f"{stage}/val_loss_val": val_losses.val , "epoch": epoch})
        # wandb.log({f"{stage}/val_loss_sum": val_losses.sum , "epoch": epoch})
        # wandb.log({f"{stage}/val_loss_count": val_losses.count , "epoch": epoch})

        wandb.log({f"{stage}/psnr_metric_avg": psnr_metric.avg , "epoch": epoch})
        wandb.log({f"{stage}/psnr_metric_val": psnr_metric.val , "epoch": epoch})
        # wandb.log({f"{stage}/psnr_metric_sum": psnr_metric.sum , "epoch": epoch})
        # wandb.log({f"{stage}/psnr_metric_count": psnr_metric.count , "epoch": epoch})

        wandb.log({f"{stage}/ssim_metric_avg": ssim_metric.avg , "epoch": epoch})
        wandb.log({f"{stage}/ssim_metric_val": ssim_metric.val , "epoch": epoch})
        # wandb.log({f"{stage}/ssim_metric_sum": ssim_metric.sum , "epoch": epoch})
        # wandb.log({f"{stage}/ssim_metric_count": ssim_metric.count , "epoch": epoch})

    def log_generator_train_image(self, epoch:int, out_images: Tensor, lr_images: Tensor, hr_images: Tensor):
        # images = wandb.Image(out_images, caption='outimatge train')
        # wandb.log({'imagesprueba', images})
        wandb.log({'generator_train': out_images, "epoch": epoch})
        # pass

    