import random

import cv2
import numpy as np
import pytest

from torch_srgan.datasets.transforms import PairedRandomCrop


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.uint8)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
@pytest.mark.parametrize("scale", [2, 4, 1/2, 1/4])
def test_paired_random_crop(interpolation, scale):
    image = get_gradient_3d(
        512, 512, (0, 0, 0), (random.randrange(128, 255), random.randrange(128, 255), random.randrange(128, 255)),
        (True, False, False)
    )
    scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)
    aug = PairedRandomCrop(128, 128, scale)
    data = aug(image=image, scaled_image=scaled_image)
    cr_image = data["image"]
    cr_scaled_image = data["scaled_image"]
    assert cr_image.shape[:2] == (128, 128)
    assert cr_scaled_image.shape[:2] == (round(128 * scale), round(128 * scale))
    rescaled_cr_image = cv2.resize(cr_image, None, fx=scale, fy=scale, interpolation=interpolation)
    difference = cv2.subtract(cr_scaled_image, rescaled_cr_image)
    b, g, r = cv2.split(difference)
    assert (b > 3).sum() == 0 and (g > 3).sum() == 0 and (r > 3).sum() == 0
