#!/usr/bin/env python3

"""
Fix up-scaling blocks name from previous versions model state dict.
"""

import argparse
import collections
import os
import torch

if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-model-path", help="Input model path", type=str)
    parser.add_argument("--out-model-path", help="Input model path", type=str)
    args = parser.parse_args()

    if args.in_model_path:
        if not os.path.exists(args.in_model_path):
            assert False, f'Checkpoint model path "{args.in_model_path} doesn\'t exists"'
    else:
        assert False, f'You need to provide a checkpoint model path'

    checkpoint = torch.load(args.in_model_path)
    model_state_dict = []

    for key, value in checkpoint["model_state_dict"].items():
        if key.startswith('upsampling_blocks.upsampling_'):
            key = '.'.join(key.split(".")[:2] + ["block"] + key.split(".")[2:])
        model_state_dict.append((key, value))

    checkpoint["model_state_dict"] = collections.OrderedDict(model_state_dict)

    torch.save(checkpoint, args.out_model_path)
