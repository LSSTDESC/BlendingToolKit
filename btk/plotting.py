"""Utility functions for plotting images."""
import numpy as np


def get_rgb(image, min_val=None, max_val=None):
    """Function to normalize 3 band input image to RGB 0-255 image.
    Args:
        image (array_like): Image array to convert to RGB image with dtype
                uint8 [bands, height, width].
        min_val (array_like): Pixel values in
        image less than or equal to this are set to zero in the RGB output.
        max_val (array_like): Pixel values in image greater than
            or equal to this are set to zero in the RGB output.
    Returns:
        uint8 array [height, width, bands] of the input image.
    """
    if image.shape[0] != 3:
        raise ValueError("Must be 3 channel in dimension 1 of image. Found {image.shape[0]}")
    if min_val is None:
        min_val = image.min(axis=-1).min(axis=-1)
    if max_val is None:
        max_val = image.max(axis=-1).max(axis=-1)
    new_image = np.transpose(image, axes=(1, 2, 0))
    new_image = (new_image - min_val) / (max_val - min_val) * 255
    new_image[new_image < 0] = 0
    new_image[new_image > 255] = 255
    return new_image.astype(np.uint8)
