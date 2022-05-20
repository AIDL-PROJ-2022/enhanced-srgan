from torch_srgan.datasets import BSDS500

def test_BSDS500_dataset():
    BSDS500_dataset = BSDS500("train", scale_factor=2, patch_size=(128, 128))
