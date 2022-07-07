import argparse
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
            assert False, f"Checkpoint model path '{args.in_model_path}' doesn't exists"
    else:
        assert False, f'You need to provide a checkpoint model path'

    checkpoint = torch.load(args.in_model_path)
    model_data = {}

    if 'epoch_i' not in checkpoint:
        assert False, f"Checkpoint file '{args.in_model_path}' is not a training checkpoint"

    # Pretraining
    if 'model_state_dict' in checkpoint:
        print(f"Extracting pre-training model data from file '{args.in_model_path}' and epoch {checkpoint['epoch_i']}")
        model_data = {
            "model_state_dict": checkpoint["model_state_dict"],
            "hparams": checkpoint["hparams"],
        }
    # Training
    elif 'g_model_state_dict' in checkpoint:
        print(f"Extracting training model data from file '{args.in_model_path}' and epoch {checkpoint['epoch_i']}")
        model_data = {
            "g_model_state_dict": checkpoint["g_model_state_dict"],
            "d_model_state_dict": checkpoint["d_model_state_dict"],
            "hparams": checkpoint["hparams"],
        }
    else:
        assert False, f"Model file '{args.in_model_path}' format is not recognized"

    torch.save(checkpoint, args.out_model_path)
