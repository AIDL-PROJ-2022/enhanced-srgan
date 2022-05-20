import random

import cv2
import numpy as np
import pytest

from torch_srgan.datasets.transforms import PairedRandomCrop, PairedCenterCrop, SimpleNormalize


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width=512, height=512, start_list=(0, 0, 0),
                    stop_list=(random.randrange(128, 255), random.randrange(128, 255), random.randrange(128, 255)),
                    is_horizontal_list=(True, False, False)):
    result = np.zeros((height, width, len(start_list)), dtype=np.uint8)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


def check_crop_and_scaled_crop(scale, patch_size, cr_image, cr_scaled_image, interpolation):
    assert cr_image.shape[:2] == patch_size
    assert cr_scaled_image.shape[:2] == (round(patch_size[0] * scale), round(patch_size[1] * scale))
    rescaled_cr_image = cv2.resize(cr_image, None, fx=scale, fy=scale, interpolation=interpolation)
    difference = cv2.subtract(cr_scaled_image, rescaled_cr_image)
    b, g, r = cv2.split(difference)
    assert (b > 3).sum() == 0 and (g > 3).sum() == 0 and (r > 3).sum() == 0


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
@pytest.mark.parametrize("scale", [2, 4, 1/2, 1/4])
@pytest.mark.parametrize("patch_size", [(32, 32), (64, 64), (128, 128)])
def test_paired_random_crop(interpolation, scale, patch_size: tuple):
    image = get_gradient_3d()
    scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)
    aug = PairedRandomCrop(*patch_size, scale)
    data = aug(image=image, scaled_image=scaled_image)
    check_crop_and_scaled_crop(scale, patch_size, data["image"], data["scaled_image"], interpolation)


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
@pytest.mark.parametrize("scale", [2, 4, 1/2, 1/4])
@pytest.mark.parametrize("patch_size", [(32, 32), (64, 64), (128, 128)])
def test_paired_center_crop(interpolation, scale, patch_size: tuple):
    image = get_gradient_3d()
    scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)
    aug = PairedCenterCrop(*patch_size, scale)
    data = aug(image=image, scaled_image=scaled_image)
    check_crop_and_scaled_crop(scale, patch_size, data["image"], data["scaled_image"], interpolation)


def test_simple_normalize():
    image = get_gradient_2d(0, 255, 512, 512, True)
    aug = SimpleNormalize()
    data = aug(image=image)
    assert np.min(data["image"]) >= 0 and np.max(data["image"]) <= 1

