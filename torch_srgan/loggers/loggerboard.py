from dis import dis
import os
import datetime
from cv2 import log
import torch
import wandb

from torchvision.utils import make_grid
# from torch_srgan.loggers.base_class import Logger
from torch.utils.tensorboard import SummaryWriter
import wandb
import git


class Loggerboard():
    def __init__(self, proj_name: str, entity_name: str, task: str,
                 generator: torch.nn.Module = None, discriminator: torch.nn.Module = None, config={}):
        logdir = os.path.join(
            "logs", f"{task}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        self.init_wandb(logdir=logdir, proj_name=proj_name, entity_name=entity_name,
                        task=task, config=config, generator=generator, discriminator=discriminator)
        self.init_tensorboard(logdir=logdir)

    def init_wandb(self, logdir: str, proj_name: str, entity_name: str, task: str, generator: torch.nn.Module = None, discriminator: torch.nn.Module = None, config={}):
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

    def init_tensorboard(self, logdir: str):
        self.writer = SummaryWriter(logdir)

    def log_stage(self, stage: str, epoch: int, train_losses: float, val_losses: float,
                  psnr_metric: float, ssim_metric: float):
        self.writer.add_scalar(f"{stage}/train_loss", train_losses, epoch)
        self.writer.add_scalar(f"{stage}/val_loss", val_losses, epoch)
        self.writer.add_scalar(f"{stage}/psnr_metric", psnr_metric, epoch)
        self.writer.add_scalar(f"{stage}/ssim_metric", ssim_metric, epoch)
        # wandb.log({f"{stage}/train_loss": train_losses}, step=epoch)
        # wandb.log({f"{stage}/val_loss": val_losses}, step=epoch)
        # wandb.log({f"{stage}/psnr_metric": psnr_metric}, step=epoch)
        # wandb.log({f"{stage}/ssim_metric": ssim_metric}, step=epoch)

    def log_image_transforms(self, epoch: int, dataset_target: str, img_transforms: dict):
        # self.writer.add_images(
        #     f"{dataset_target}/img_transforms", img_transforms, epoch)
        wandb.log(
            {f"{dataset_target}/img_transforms": img_transforms, "epoch": epoch})

    def log_images(self, epoch: int, target: str, lr_images: torch.Tensor,
                   out_images: torch.Tensor, gt_images: torch.Tensor):
        # for idx, out_image in enumerate(out_images):
        #     self.writer.add_image(
        #         f'{target}/image-{idx}', make_grid([out_image, gt_images[idx]]), epoch)

        images = []
        for idx, out_image in enumerate(out_images):
            images.append(make_grid([out_image, gt_images[idx]]))

        wandb.log({f'{target}/images': [wandb.Image(im)
                  for im in images], "epoch": epoch})

    def log_model_graph(self, model, train_loader):
        batch, _ = next(iter(train_loader))
        # TODO: This could be an error and have to be checked and deleted this line
        batch = batch.cuda() if torch.cuda.is_available() else batch
        self.writer.add_graph(model, batch)
