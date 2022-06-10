import cv2
import matplotlib.pyplot as plt

from torch_srgan.datasets import BSDS500


def visualize(image):
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def test_BSDS500_dataset():
    dataset = BSDS500("train", scale_factor=3, patch_size=(129, 129))
    assert len(dataset) == 200
    lr_img, hr_img = dataset[0]
    visualize(lr_img.permute(1, 2, 0))
    visualize(hr_img.permute(1, 2, 0))

