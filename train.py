"""
ESRGAN Network training script.
"""

import argparse
import os
import warnings
from typing import List

import albumentations
import torch.nn as nn
import torch
import piq

from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_srgan.datasets import BSDS500
from torch_srgan.loggers.loggerboard import Loggerboard
from torch_srgan.models.esrgan import GeneratorESRGAN


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def supervised_stage_train(loss_f: nn.Module, optimizer: torch.optim.Optimizer, train_losses: AverageMeter, epoch: int):
    # Switch generator model to train mode
    generator.train()

    # Iterate over train image batches for this epoch
    for i, (lr_images, hr_images) in enumerate(tqdm(train_dataloader, desc="[TRAINING]", leave=False, ascii=True)):

        # Move images to device
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        # Set optimizer gradients to zero
        optimizer.zero_grad()

        # Generate a high resolution images from low resolution input
        out_images = generator(lr_images)

        # Measure pixel-wise content loss against ground truth image
        loss = loss_f(out_images, hr_images)
        train_losses.update(loss.item(), lr_images.size(0))

        # Backpropagate gradients and go to next optimizer step
        loss.backward()
        optimizer.step()

        # Log processed images and results
        if epoch == 1 or epoch % 10 == 0:
            # logger.log_image_transforms(epoch, "train", transforms)
            logger.log_images(epoch, "train", lr_images, out_images, hr_images)


def supervised_stage_validate(val_losses: AverageMeter, psnr_metric: AverageMeter, ssim_metric: AverageMeter,
                              epoch: int):
    # Switch generator model to evaluation mode
    generator.eval()

    # Disable gradient propagation
    with torch.no_grad():
        # Iterate over validation image batches for this epoch
        for i, (lr_images, hr_images) in enumerate(tqdm(val_dataloader, desc="[VALIDATION]", leave=False, ascii=True)):

            # Move images to device
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            # Generate a high resolution images from low resolution input
            out_images = generator(lr_images)

            # Measure pixel-wise content loss against ground truth image
            loss = content_loss(out_images, hr_images)
            val_losses.update(loss.item(), lr_images.size(0))
            # Measure PSNR metric against ground truth image
            psnr = piq.psnr(hr_images, out_images, data_range=1.0, reduction="mean", convert_to_greyscale=False)
            psnr_metric.update(psnr.item(), lr_images.size(0))
            # Measure SSIM metric against ground truth image
            ssim, _ = piq.ssim(
                hr_images, out_images, kernel_size=11, kernel_sigma=1.5, k1=0.01, k2=0.03,
                data_range=1.0, reduction="mean", full=True
            )
            ssim_metric.update(ssim.item(), lr_images.size(0))

            # Log processed images and results
            if epoch == 1 or epoch % 10 == 0:
                logger.log_images(epoch, "validation", lr_images, out_images, hr_images)


def exec_supervised_stage(num_epoch: int, lr: float, sched_step: int, sched_gamma: float, train_aug_transforms: List,
                          loss_f: nn.Module = nn.L1Loss, start_epoch: int = 0, store_checkpoint: bool = True):

    # Define optimizer and scheduler for supervised training stage
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)

    # Set train dataset transforms
    train_img_dataset.set_dataset_transforms(train_aug_transforms)

    # Define metrics
    train_losses = AverageMeter("Content loss [Train]:", ":.4e")
    val_losses = AverageMeter("Content loss [Valid]:", ":.4e")
    psnr_metric = AverageMeter("PSNR:", ":.4f")
    ssim_metric = AverageMeter("SSIM:", ":.4f")

    # If start epoch is bigger than 0, load model state from local storage
    if start_epoch > 0:
        checkpoint = torch.load(f"saved_models/generator_1_{start_epoch}.pt")
        generator.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Train model for specified number of epoch
    for epoch in tqdm(range(start_epoch+1, num_epoch+1), desc="[STAGE 1]"):

        # Reset metrics
        train_losses.reset()
        val_losses.reset()
        psnr_metric.reset()
        ssim_metric.reset()

        # Train model
        supervised_stage_train(loss_f, optimizer, train_losses, epoch)
        # Validate model
        supervised_stage_validate(val_losses, psnr_metric, ssim_metric, epoch)

        # Print metrics after this epoch
        tqdm.write(
            f"[Epoch {epoch}/{num_epoch}] METRICS:\r\n"
            f"  - {str(train_losses)}\r\n"
            f"  - {str(val_losses)}\r\n"
            f"  - {str(psnr_metric)}\r\n"
            f"  - {str(ssim_metric)}\r\n"
        )

        # log stage
        logger.log_stage('stage-1', epoch, train_losses.avg, val_losses.avg, psnr_metric.avg, ssim_metric.avg)

        # Perform scheduler step
        scheduler.step()

        # Store model parameters
        if store_checkpoint:
            checkpoint = {
                "model_state_dict": generator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }
            torch.save(checkpoint, f"saved_models/generator_1_{epoch}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-epoch", type=int, default=0, help="epoch to start training from")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_cpu = os.cpu_count()

    # Enable cudnn benchmarking if available
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    # Create saved models directory if not exist
    os.makedirs("saved_models", exist_ok=True)

    # Ignore warnings
    warnings.filterwarnings('ignore')

    # Define hyper-parameters
    hparams = {
        "scale_factor": 4,
        "cr_patch_size": (128, 128),
        "batch_size": 16,
        "img_channels": 3,
        "training": [
            {
                "num_epoch": 10,
                "lr": 0.0002,
                "sched_step": 500,
                "sched_gamma": 0.5
            }
        ],
        "generator": {
            "rrdb_channels": 64,
            "growth_channels": 32,
            "num_basic_blocks": 16,
            "num_dense_blocks": 3,
            "num_residual_blocks": 5,
            "residual_scaling": 0.2
        }
    }

    # Define train dataset augmentation transforms
    spatial_transforms: List = [
        albumentations.OneOf(
            [
                albumentations.Flip(p=0.75),        # p = 1/4 (vflip) + 1/4 (hflip) + 1/4 (flip)
                albumentations.Transpose(p=0.25)    # p = 1/4
            ],
            p=0.5
        )
    ]
    hard_transforms: List = [
        albumentations.CoarseDropout(max_holes=8, max_height=2, max_width=2, p=0.5),
        albumentations.ImageCompression(quality_lower=65, p=0.25)
    ]

    # Define dataset class
    dataset_class = BSDS500
    # Define train, validation and test datasets to use
    train_img_dataset = dataset_class(
        target='train', scale_factor=hparams["scale_factor"], patch_size=hparams["cr_patch_size"],
        # retrieve_transforms_info=True
    )
    val_img_dataset = dataset_class(
        target='val', scale_factor=hparams["scale_factor"], patch_size=hparams["cr_patch_size"], download=False
    )
    test_img_dataset = dataset_class(
        target='test', scale_factor=hparams["scale_factor"], patch_size=hparams["cr_patch_size"], download=False
    )
    # Define also the dataloaders
    train_dataloader = DataLoader(
        train_img_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=n_cpu, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_img_dataset, batch_size=hparams["batch_size"], num_workers=n_cpu, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_img_dataset, batch_size=hparams["batch_size"], num_workers=n_cpu, pin_memory=True
    )

    # Initialize generator and discriminator models
    generator = GeneratorESRGAN(
        img_channels=hparams["img_channels"], scale_factor=hparams["scale_factor"], **hparams["generator"]
    ).to(device)
    # TODO: DEFINE DISCRIMINATOR

    # Logger initialize
    logger = Loggerboard(proj_name='ESRGAN', entity_name="esrgan-aidl-2022", task='training', generator=generator, config=hparams)
    logger.log_model_graph(model=generator, train_loader=train_dataloader)

    # Define loss functions
    content_loss = nn.L1Loss().to(device)

    #######################
    #  Training (STAGE 1) #
    #######################

    # Write empty newline
    print()

    # Execute supervised pre-training stage
    exec_supervised_stage(
        **hparams["training"][0], train_aug_transforms=(spatial_transforms + hard_transforms), loss_f=content_loss,
        start_epoch=opt.start_epoch
    )
