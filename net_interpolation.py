#!/usr/bin/env python3

"""
ESRGAN Network interpolation script.
"""

import argparse
import os
import torch

from collections import OrderedDict


def load_model(model_path: str):
    if model_path:
        if not os.path.exists(model_path):
            assert False, f"Model data file path '{model_path}' doesn't exists"
    else:
        assert False, f'You need to provide a model data file path'

    model_file_data = torch.load(model_path)
    model_state_dict = model_file_data.get("model_state_dict", model_file_data.get("g_model_state_dict", None))

    if model_state_dict is None:
        assert False, f"Provided model data file path '{model_path}' has an unknown format"

    return model_state_dict, model_file_data["hparams"]


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--psnr-model-path", help="PSNR-based (pre-training) input model path", type=str, required=True)
    parser.add_argument("--esrgan-model-path", help="ESRGAN-based (training) input model path", type=str, required=True)
    parser.add_argument("--alpha", help="Network interpolation alpha", type=float, required=True)
    parser.add_argument("out_model_path", help="Output interpolated model", type=str)
    args = parser.parse_args()

    if args.psnr_model_path:
        if not os.path.exists(args.psnr_model_path):
            assert False, f"PSNR model path '{args.psnr_model_path}' doesn't exists"
    else:
        assert False, f'You need to provide a PSNR model path'

    if args.esrgan_model_path:
        if not os.path.exists(args.esrgan_model_path):
            assert False, f"ESRGAN model path '{args.esrgan_model_path}' doesn't exists"
    else:
        assert False, f'You need to provide a ESRGAN model path'

    if args.alpha < 0 or args.alpha > 1:
        assert False, f'Network interpolation alpha must be in the range [0, 1]'

    alpha = args.alpha

    # Load models data
    psnr_model_state_dict, _ = load_model(args.psnr_model_path)
    esrgan_model_state_dict, hparams = load_model(args.esrgan_model_path)

    if psnr_model_state_dict.keys() != esrgan_model_state_dict.keys():
        assert False, "Generator parameters from PSNR and ESRGAN trainings must be equal!"

    net_interp_model_state_dict = OrderedDict()
    # Interpolate parameter values
    for key, val_psnr in psnr_model_state_dict.items():
        val_esrgan = esrgan_model_state_dict[key]
        net_interp_model_state_dict[key] = (1 - alpha) * val_psnr + alpha * val_esrgan

    # Define interpolated model data dict
    model_data = {
        "model_state_dict": net_interp_model_state_dict,
        "stage": "interpolation",
        "alpha": alpha,
        "hparams": hparams
    }

    torch.save(model_data, args.out_model_path)
