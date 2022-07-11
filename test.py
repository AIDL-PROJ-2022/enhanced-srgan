"""
ESRGAN Network test script.
"""

import argparse
import os
import pathlib
import matplotlib.figure
import numpy as np
import torch
import piq
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset
from torch.nn import functional as F
from tqdm import tqdm
from typing import List, Tuple

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


def validate_model(best_plots_to_show: int = 0, out_dir: str = None):
    print()
    print(">>> Executing model testing")
    print()
    print("-" * 64)
    print()

    # Enable plot auto-layout
    plt.rcParams["figure.autolayout"] = True
    # Acquire default dots per inch value of matplotlib
    dpi = matplotlib.rcParams['figure.dpi']
    # Define margin in pixels
    margin = 50

    # Initialize an array containing the plot and the PSNR value
    result_figues: List[Tuple[float, matplotlib.figure.Figure]] = []

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
            c_loss = content_loss(out_image, hr_image).item()
            content_loss_metric.update(c_loss)

            # Measure perceptual loss against ground truth image (VGG-based loss)
            p_loss = perceptual_loss(out_image, hr_image).item()
            perceptual_loss_metric.update(p_loss)

            # Measure PSNR metric against ground truth image
            psnr = piq.psnr(hr_image, out_image, data_range=1.0, reduction="mean", convert_to_greyscale=False).item()
            psnr_metric.update(psnr)

            # Measure SSIM metric against ground truth image
            ssim, _ = piq.ssim(hr_image, out_image, data_range=1.0, reduction="mean", full=True)
            ssim = ssim.item()
            ssim_metric.update(ssim)

        # Create image result plot
        # Upscale LR image
        lr_image = F.interpolate(lr_image, scale_factor=hparams["scale_factor"], mode='nearest-exact')
        # Define an array containing the images with its title
        images = [
            ("Low Resolution", tensor_to_image(lr_image)),
            ("Super Resolved", tensor_to_image(out_image)),
            ("Ground Truth", tensor_to_image(hr_image))
        ]
        # Retrieve the image height and width
        height, width, _ = images[0][1].shape
        # Define figure size and left and bottom margins
        figsize_w = 3 * ((width + 2 * margin) / dpi)
        figsize_h = (height + 2 * margin) / dpi
        # Axes ratio
        left = margin / dpi / figsize_w
        bottom = margin / dpi / figsize_h
        # Add extra space for metrics text
        figsize_h += (2 * margin) / dpi
        # Create a new plot figure
        fig = plt.figure(figsize=(figsize_w, figsize_h))
        # Adjust figure subplots positions
        fig.subplots_adjust(left=left, bottom=bottom, right=(1. - left), top=(1. - bottom))
        # Add it to the plot
        for j, (title, img) in enumerate(images, start=1):
            ax = fig.add_subplot(1, 3, j)
            ax.imshow(img)
            ax.set_title(title, fontdict=dict(fontsize=16, fontweight='bold'))
        # Add metrics to the plot
        fig.text(
            0.5, 0.01, f"PSNR: {round(psnr, 2)}db / SSIM: {round(ssim, 2)}",
            ha='center', va='bottom', fontsize=16, fontweight='bold'
        )
        # Append figure to result figures array
        result_figues.append((psnr, fig))
        # If user specified an output directory, store the result plot there
        if out_dir is not None:
            fig.savefig(os.path.join(out_dir, f'{i:05d}.png'), bbox_inches='tight')

    # Log metrics
    print(
        f"\r\nVALIDATION METRICS:\r\n"
        f"  - {str(content_loss_metric)}\r\n"
        f"  - {str(perceptual_loss_metric)}\r\n"
        f"  - {str(psnr_metric)}\r\n"
        f"  - {str(ssim_metric)}\r\n"
    )
    print("-" * 64)

    # Sort all results from its PSNR value
    result_figues.sort(key=lambda x: x[0])
    # Hide all plots with bad results
    for _, fig in result_figues[best_plots_to_show:]:
        plt.close(fig)

    # Show all plots
    plt.show()


def debugger_is_active() -> bool:
    import sys
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda: None)
    return gettrace() is not None


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--datasets",
        help="Datasets to use to test the specified model. "
             "Datasets need to be specified sepparated by a coma. Example: --datasets=set5,set14. "
             "Available values are: 'div2k', 'bsds500', 'set5', and 'set14'",
        type=str
    )
    parser.add_argument(
        "-s", "--show-results", help="Show N best PSNR of all tested images",
        type=int, default=0
    )
    parser.add_argument(
        "-o", "--out-dir", help="Specify output directory where all results will be stored",
        type=pathlib.Path, default=None
    )
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

    test_datasets = []
    # Define datasets to use
    for dataset_name in args.datasets.split(","):
        # Define dataset object from its name
        if dataset_name == "bsds500":
            dataset = datasets.BSDS500(target='test', scale_factor=hparams["scale_factor"], patch_size=None)
        elif dataset_name == "div2k":
            dataset = datasets.DIV2K(target='test', scale_factor=hparams["scale_factor"], patch_size=(512, 512))
        elif dataset_name == "set5":
            dataset = datasets.Set5(scale_factor=hparams["scale_factor"])
        elif dataset_name == "set14":
            dataset = datasets.Set14(scale_factor=hparams["scale_factor"])
        else:
            raise ValueError(f"Unrecognized dataset name: {dataset_name}")
        # Append dataset to the test datasets array
        test_datasets.append(dataset)

    # Define a concatenated dataset containing all images from specified datasets
    test_dataset = ConcatDataset(test_datasets)

    # Define data loader
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=os.cpu_count() - 1, shuffle=False, pin_memory=True
    )

    # Make sure that output directory is created
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    validate_model(args.show_results, args.out_dir)
