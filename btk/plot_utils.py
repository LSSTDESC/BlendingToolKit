import numpy as np
import matplotlib.pyplot as plt


def get_rgb(image, min_val=None, max_val=None):
    """Function to normalize 3 band input image to RGB 0-255 image.

    Args:
        image (array_like): Image array to convert to RGB image with dtype
                uint8 [bands, height, width].
        min_val (float32 or 3-d float array, default=`None`): Pixel values in
        image less than or equal to this are set to zero in the RGB output.
        max_val (float32, default=`None`): Pixel values in image greater than
            or equal to this are set to zero in the RGB output.

    Returns:
        uint8 array [height, width, bands] of the input image.
    """
    if image.shape[0] != 3:
        raise ValueError("Must be 3 channel in dimension 1 of image")
    if min_val is None:
        min_val = image.min(axis=-1).min(axis=-1)
    if max_val is None:
        max_val = image.max(axis=-1).max(axis=-1)
    new_image = np.transpose(image, axes=(1, 2, 0))
    new_image = (new_image - min_val) / (max_val - min_val)*255
    new_image[new_image < 0] = 0
    new_image[new_image > 255] = 255
    return new_image.astype(np.uint8)


def get_rgb_image(image, normalize_with_image=None):
    """Returns RGB (0-255) image corresponding to the input 3 band image.

    If scarlet.display is imported then the normalization is performed by
    scarlet Asinh function. If not, a basic normalization is performed.

    Args:
        image : Image array (float32) to convert to RGB [bands, height, width].
        normalize_with_image: Image array (float32) to normalize input image
            with [bands, height, width].

    Returns:
        uint8 array [height, width, bands] of the input image.
    """
    try:
        import scarlet
        if normalize_with_image:
            norm = scarlet.display.Asinh(img=normalize_with_image, Q=20)
        else:
            norm = None
        img_rgb = scarlet.display.img_to_rgb(image, norm=norm)
    except ImportError:
        # scarlet not installed, basic normalize image to 0-255
        if normalize_with_image is None:
            min_val = None
            max_val = None
        else:
            min_val = np.min(normalize_with_image, axis=1).min(axis=-1)
            max_val = np.max(normalize_with_image, axis=1).max(axis=-1)
        img_rgb = get_rgb(image, min_val=min_val, max_val=max_val)
    return img_rgb


def plot_blends(blend_images, blend_list, band_indices=[1, 2, 3],
                detected_centers=None, limits=None):
    """Plots blend images as RGB image, sum in all bands, and RGB image with
    centers of objects marked.

    Outputs of btk draw are plotted here. Blend_list must contain true  centers
    of the objects. If detected_centers are input, then the centers are also
    shown in the third panel along with the true centers.

    Args:
        image (array_like): Image array to plot [batch, height, width, bands].
        blend_list (list) : List of `astropy.table.Table` with entries of true
            objects. Length of list must be the batch size.
        band_indices (list, default=[1,2,3]): list of length 3 with indices of
            bands that are to be plotted in the RGB image.
        detected_centers (list, default=`None`): List of `numpy.ndarray` or
            lists with centers of detected centers for each image in batch.
            Length of list must be the batch size. Each list entry must be a
            list or `numpy.ndarray` of dimensions [N, 2].
        limits(list, default=`None`): List of start and end coordinates to
            display image within. Note: limits are applied to both height and
            width dimensions.
    """
    batch_size = len(blend_list)
    if detected_centers is None:
        detected_centers = [[]]*batch_size
    for i in range(batch_size):
        num = len(blend_list[i])
        images = np.transpose(blend_images[i, :, :, 1:4], axes=(2, 0, 1))
        blend_img_rgb = get_rgb_image(images)
        _, ax = plt.subplots(1, 3, figsize=(8, 3))
        ax[0].imshow(blend_img_rgb)
        if limits:
            ax[0].set_xlim(limits)
            ax[0].set_ylim(limits)
        ax[0].set_title("gri bands")
        ax[0].axis('off')
        ax[1].imshow(np.sum(blend_images[i, :, :, :], axis=2))
        ax[1].set_title("Sum")
        if limits:
            ax[1].set_xlim(limits)
            ax[1].set_ylim(limits)
        ax[1].axis('off')
        ax[2].imshow(blend_img_rgb)
        ax[2].set_title(f"{num} objects with centers")
        for entry in blend_list[i]:
            ax[2].plot(entry['dx'], entry['dy'], 'rx')
        if limits:
            ax[2].set_xlim(limits)
            ax[2].set_ylim(limits)
        for cent in detected_centers[i]:
            ax[2].plot(cent[0], cent[1], 'go', fillstyle='none', ms=10, mew=2)
        ax[2].axis('off')
    plt.show()
