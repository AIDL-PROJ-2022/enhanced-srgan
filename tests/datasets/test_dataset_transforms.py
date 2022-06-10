import cv2
import numpy as np
import pytest
import matplotlib.pyplot as plt
import os

from PIL import Image
from albumentations.augmentations.crops import functional as F
from torch_srgan.datasets.transforms import PairedRandomCrop, PairedCenterCrop, SimpleNormalize


def open_test_image(image_name="test_image.png"):
    base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../assets")
    img_path = os.path.join(base_path, image_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def adj_image_size(image, scale):
    img_h, img_w = image.shape[:2]
    img_h -= img_h % scale
    img_w -= img_w % scale
    return F.center_crop(image, img_h, img_w)


def check_crop_and_scaled_crop(scale, patch_size, cr_image, cr_scaled_image, interpolation):
    lr_patch_size = ((patch_size[0] // scale), (patch_size[1] // scale))
    assert cr_image.shape[:2] == lr_patch_size
    assert cr_scaled_image.shape[:2] == patch_size
    visualize(cr_image)
    visualize(cr_scaled_image)
    rescaled_cr_image = cv2.resize(cr_scaled_image, dsize=lr_patch_size, interpolation=interpolation)
    difference = cv2.subtract(cr_image, rescaled_cr_image)
    b, g, r = cv2.split(difference)
    assert cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0


def visualize(image):
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.show()


@pytest.mark.parametrize("interpolation", [cv2.INTER_CUBIC])
#@pytest.mark.parametrize("scale", [2, 4, 8])
@pytest.mark.parametrize("scale", [3])
# @pytest.mark.parametrize("patch_size", [(96, 96), (128, 128), (192, 192)])
@pytest.mark.parametrize("patch_size", [(128, 128)])
def test_paired_random_crop(interpolation, scale, patch_size: tuple):
    image = open_test_image()
    image = adj_image_size(image, scale)
    hr_img_h, hr_img_w = image.shape[:2]
    print(image.shape[:2])
    lr_img_size = (hr_img_w // scale, hr_img_h // scale)
    print(lr_img_size)
    lr_image = cv2.resize(image, lr_img_size, interpolation=interpolation)
    aug = PairedRandomCrop(patch_size, scale)
    data = aug(image=lr_image, scaled_image=image)
    check_crop_and_scaled_crop(scale, patch_size, data["image"], data["scaled_image"], interpolation)


@pytest.mark.parametrize("interpolation", [cv2.INTER_AREA])
#@pytest.mark.parametrize("scale", [2, 4, 8])
@pytest.mark.parametrize("scale", [3])
# @pytest.mark.parametrize("patch_size", [(96, 96), (128, 128), (192, 192)])
@pytest.mark.parametrize("patch_size", [(129, 129)])
def test_paired_center_crop(interpolation, scale, patch_size: tuple):
    image = open_test_image()
    image = adj_image_size(image, scale)
    lr_resize_scale = 1.0 / float(scale)
    lr_image = cv2.resize(image, None, fx=lr_resize_scale, fy=lr_resize_scale, interpolation=interpolation)
    aug = PairedCenterCrop(patch_size, scale)
    data = aug(image=lr_image, scaled_image=image)
    check_crop_and_scaled_crop(scale, patch_size, data["image"], data["scaled_image"], interpolation)


def test_simple_normalize():
    image = open_test_image()
    aug = SimpleNormalize()
    data = aug(image=image)
    assert np.min(data["image"]) >= 0 and np.max(data["image"]) <= 1

