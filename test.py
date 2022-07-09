"""
ESRGAN Network test script.
"""

import argparse
import os
import numpy as np
import torch
import piq
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.nn import functional as F
from tqdm import tqdm

import torch_srgan.datasets as datasets
from torch_srgan.models.RRDBNet import RRDBNet
from torch_srgan.nn.criterions import ContentLoss, PerceptualLoss


class AverageMeter(object):
    """
    Computes and stores the average, maximum and minimum value.
    """

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.avg = 0
        self.max = None
        self.min = None
        self._sum = 0
        self._count = 0

    def reset(self):
        self.avg = 0
        self.max = None
        self.min = None
        self._sum = 0
        self._count = 0

    def update(self, val: float, n: int = 1):
        # Compute the new maximum and minimum value
        self.max = max(val, self.max) if self.max is not None else val
        self.min = min(val, self.min) if self.min is not None else val
        # Compute the new average value
        self._sum += val * n
        self._count += n
        self.avg = self._sum / self._count

    def __str__(self):
        fmtstr = "{name}: {avg" + self.fmt + "} (Min: {min" + self.fmt + "} / Max: {max" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def tensor_to_image(out_image: torch.Tensor) -> np.array:
    # Make sure the image is between the range [0, 1]
    out_image = torch.clamp(out_image, min=0, max=1)
    # Convert the image to a numpy array again
    out_image = out_image.squeeze().float().cpu().numpy()
    # Re-arrange image and de-normalize it
    out_image = out_image.transpose((1, 2, 0))
    out_image = (out_image * 255.0).round().astype(np.uint8)

    return out_image


def validate_model(image_plot_interval: int = 50):
    print()
    print(">>> Executing model testing")
    print()
    print("-" * 64)
    print()

    plt.rcParams["figure.autolayout"] = True

    plot_cols = 2
    plot_rows = int((len(bsds500_test_dataset) / image_plot_interval) / plot_cols)

    # Create plot figure
    _, plot_axs = plt.subplots(nrows=plot_rows, ncols=plot_cols)

    # Iterate over test images. Batch size is configured to 1, so only one image will be processed each time
    for i, (lr_image, hr_image) in enumerate(tqdm(test_dataloader, desc="[VALIDATION]")):
        # Move images to device
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)

        # Generate a high resolution images from low resolution input
        with torch.no_grad():
            out_image = generator(lr_image)
            # Make sure that images are between the range [0, 1]
            out_image = torch.clamp(out_image, min=0, max=1)

            # Measure pixel-wise content loss against ground truth image (Pixel-wise loss)
            c_loss = content_loss(out_image, hr_image)
            content_loss_metric.update(c_loss.item())

            # Measure perceptual loss against ground truth image (VGG-based loss)
            p_loss = perceptual_loss(out_image, hr_image)
            perceptual_loss_metric.update(p_loss.item())

            # Measure PSNR metric against ground truth image
            psnr = piq.psnr(hr_image, out_image, data_range=1.0, reduction="mean", convert_to_greyscale=False)
            psnr_metric.update(psnr.item())

            # Measure SSIM metric against ground truth image
            ssim, _ = piq.ssim(
                hr_image, out_image, kernel_size=11, kernel_sigma=1.5, k1=0.01, k2=0.03,
                data_range=1.0, reduction="mean", full=True
            )
            ssim_metric.update(ssim.item())

        # Add image to plot
        if i % image_plot_interval == 0:
            # Upscale LR image
            lr_image = F.interpolate(lr_image, scale_factor=hparams["scale_factor"], mode='nearest-exact')
            # Create a grid with the three images
            grid_img = make_grid([lr_image.squeeze(), out_image.squeeze(), hr_image.squeeze()])
            # Convert it to a numpy array
            grid_img_np = tensor_to_image(grid_img)
            # Add it to the plot
            plot_i = int(i / image_plot_interval)
            row = int(plot_i / plot_cols)
            col = int(plot_i % plot_cols)
            plot_axs[row, col].axis('off')
            plot_axs[row, col].set_title("LR / GEN / GT")
            plot_axs[row, col].imshow(grid_img_np)

    # Log metrics
    print(
        f"\r\nVALIDATION METRICS:\r\n"
        f"  - {str(content_loss_metric)}\r\n"
        f"  - {str(perceptual_loss_metric)}\r\n"
        f"  - {str(psnr_metric)}\r\n"
        f"  - {str(ssim_metric)}\r\n"
    )
    print()
    print("-" * 64)

    # Show images
    plt.show()


def debugger_is_active() -> bool:
    import sys
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda: None)
    return gettrace() is not None


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="ESRGAN model path to test", type=str)
    args = parser.parse_args()

    # Load model data and retrieve state dict and hyper parameters
    model_file_data = torch.load(args.model_path)
    model_state_dict = model_file_data.get("model_state_dict", model_file_data.get("g_model_state_dict", None))
    hparams = model_file_data.get("hparams")

    if model_state_dict is None or hparams is None:
        assert False, f"Provided model data file path '{args.model_path}' has an unknown format"

    # Define device to use for inference
    cuda_available = torch.cuda.is_available() and not debugger_is_active()
    # cuda_available = False
    device = torch.device("cuda" if cuda_available else "cpu")

    # Define generator model
    generator = RRDBNet(
        img_channels=hparams["img_channels"], scale_factor=hparams["scale_factor"], **hparams["generator"]
    )
    # Load generator model parameters
    generator.load_state_dict(model_state_dict)

    # Define losses used during validation
    content_loss = ContentLoss(**hparams["content_loss"])
    perceptual_loss = PerceptualLoss(**hparams["perceptual_loss"])

    # Move everything to device
    generator.to(device)
    content_loss.to(device)
    perceptual_loss.to(device)

    # Set generator to evaluation mode
    generator.eval()

    # Define metrics
    content_loss_metric = AverageMeter("Generator Content Loss", ":.4e")
    perceptual_loss_metric = AverageMeter("Generator Perceptual Loss", ":.4e")
    psnr_metric = AverageMeter("PSNR", ":.4f")
    ssim_metric = AverageMeter("SSIM", ":.4f")

    # Define datasets to use:
    # BSDS500
    bsds500_test_dataset = datasets.BSDS500(target='test', scale_factor=hparams["scale_factor"], patch_size=None)

    # Define data loader
    test_dataloader = DataLoader(
        bsds500_test_dataset, batch_size=1, num_workers=os.cpu_count() - 1, shuffle=False, pin_memory=True
    )

    validate_model()
