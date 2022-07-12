#!/usr/bin/env python3

"""
RRDB Network inference script.
"""

import argparse
import glob
import os
import cv2
import numpy as np
import torch

from pathlib import Path

from torch_sr.models import RRDBNet
from torch_sr.nn import ContentLoss, PerceptualLoss


def debugger_is_active() -> bool:
    import sys
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda: None)
    return gettrace() is not None


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-path", help="Input generator model path. Can be a checkpoint or a final model",
        type=str, required=True
    )
    parser.add_argument(
        "-o", "--out-dir", help="Output directory where inferred images will be stored",
        type=str, default=None
    )
    parser.add_argument('images', type=str, nargs='+')
    args = parser.parse_args()

    if args.model_path:
        if not os.path.exists(args.model_path):
            assert False, f"Checkpoint model path '{args.model_path}' doesn't exists"
    else:
        assert False, f'You need to provide a checkpoint model path'

    # Load model data and retrieve state dict and hyper parameters
    model_file_data = torch.load(args.model_path)
    model_state_dict = model_file_data.get("model_state_dict", model_file_data.get("g_model_state_dict", None))
    hparams = model_file_data.get("hparams")

    if model_state_dict is None or hparams is None:
        assert False, f"Provided model data file path '{args.model_path}' has an unknown format"

    # Define device to use for inference
    cuda_available = torch.cuda.is_available() and not debugger_is_active()
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

    # Move model to device and set it to evaluation mode
    generator.to(device)
    generator.eval()

    # Parse image paths
    full_paths = [os.path.join(os.getcwd(), path) for path in args.images]
    image_files = set()
    for path in full_paths:
        if os.path.isfile(path):
            image_files.add(path)
        else:
            image_files |= set(glob.glob(path + '/*' + args.extension))

    # Make sure that output directory is created
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # Iterate over all given images
    for image_f in image_files:
        # Get image file path object
        image_file_path = Path(image_f)
        # Load image using OpenCV
        image = cv2.imread(image_f, cv2.IMREAD_COLOR)
        # Normalize image
        image = image.astype(np.float32) * 1.0 / 255.0
        # Convert image to Tensor
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image.unsqueeze(0)
        # Move image to device
        image = image.to(device)
        # Pass it through the network
        with torch.no_grad():
            out_image = generator(image)
        # Make sure the image is between the range [0, 1]
        out_image = torch.clamp(out_image, min=0, max=1)
        # Convert the image to a numpy array again
        out_image = out_image.squeeze().float().cpu().numpy()
        # Re-arrange image and de-normalize it
        out_image = out_image.transpose((1, 2, 0))
        out_image = (out_image * 255.0).round().astype(np.uint8)
        # Generate output image file name
        out_img_file_name = image_file_path.stem + "_hr" + image_file_path.suffix
        # Define output image directory
        if args.out_dir:
            out_image_path = os.path.join(args.out_dir, out_img_file_name)
        else:
            out_image_path = os.path.join(image_file_path.parent, out_img_file_name)
        # Write output image
        cv2.imwrite(out_image_path, out_image)
