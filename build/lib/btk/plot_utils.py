import numpy as np
import matplotlib.pyplot as plt
import btk
import matplotlib.patches as patches


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
        raise ValueError("Must be 3 channel in dimension 1 of image"
                         f"Found {image.shape[0]}")
    if min_val is None:
        min_val = image.min(axis=-1).min(axis=-1)
    if max_val is None:
        max_val = image.max(axis=-1).max(axis=-1)
    new_image = np.transpose(image, axes=(1, 2, 0))
    new_image = (new_image - min_val) / (max_val - min_val) * 255
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
        import scarlet.display
        if normalize_with_image is not None:
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


def plot_blends(blend_images, blend_list, detected_centers=None,
                limits=None, band_indices=[1, 2, 3]):
    """Plots blend images as RGB image, sum in all bands, and RGB image with
    centers of objects marked.

    Outputs of btk draw are plotted here. Blend_list must contain true  centers
    of the objects. If detected_centers are input, then the centers are also
    shown in the third panel along with the true centers.

    Args:
        blend_images (array_like): Array of blend scene images to plot
            [batch, height, width, bands].
        blend_list (list) : List of `astropy.table.Table` with entries of true
            objects. Length of list must be the batch size.
        detected_centers (list, default=`None`): List of `numpy.ndarray` or
            lists with centers of detected centers for each image in batch.
            Length of list must be the batch size. Each list entry must be a
            list or `numpy.ndarray` of dimensions [N, 2].
        limits(list, default=`None`): List of start and end coordinates to
            display image within. Note: limits are applied to both height and
            width dimensions.
        band_indices (list, default=[1,2,3]): list of length 3 with indices of
            bands that are to be plotted in the RGB image.
    """
    batch_size = len(blend_list)
    if len(band_indices) != 3:
        raise ValueError(f"band_indices must be a list with 3 entries, not \
            {band_indices}")
    if detected_centers is None:
        detected_centers = [[]] * batch_size
    if (len(detected_centers) != batch_size or
            blend_images.shape[0] != batch_size):
        raise ValueError(f"Length of detected_centers and length of blend_list\
            must be equal to first dimension of blend_images, found \
            {len(detected_centers), len(blend_list), len(blend_images)}")
    for i in range(batch_size):
        num = len(blend_list[i])
        images = np.transpose(blend_images[i],
                              axes=(2, 0, 1))
        blend_img_rgb = get_rgb_image(images[band_indices])
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


def plot_with_isolated(blend_images, isolated_images, blend_list,
                       detected_centers=None, limits=None,
                       band_indices=[1, 2, 3]):
    """Plots blend images and isolated images of all objects in the blend as
    RGB images.

    Outputs of btk draw are plotted here. Blend_list must contain true  centers
    of the objects. If detected_centers are input, then the centers are also
    shown in the third panel along with the true centers.

    Args:
        blend_images(array_like): Array of blend scene images to plot
            [batch, height, width, bands].
        isolated_images (array_like): Array of isolated object images to plot
            [batch, max number of objects, height, width, bands].
        blend_list(list) : List of `astropy.table.Table` with entries of true
            objects. Length of list must be the batch size.
        detected_centers(list, default=`None`): List of `numpy.ndarray` or
            lists with centers of detected centers for each image in batch.
            Length of list must be the batch size. Each list entry must be a
            list or `numpy.ndarray` of dimensions [N, 2].
        limits(list, default=`None`): List of start and end coordinates to
            display image within. Note: limits are applied to both height and
            width dimensions.
        band_indices (list, default=[1,2,3]): list of length 3 with indices of
            bands that are to be plotted in the RGB image.
    """
    b_size = len(blend_list)
    if len(band_indices) != 3:
        raise ValueError(f"band_indices must be a list with 3 entries, not \
            {band_indices}")
    if detected_centers is None:
        detected_centers = [[]] * b_size
    if (len(detected_centers) != b_size or len(isolated_images) != b_size or
            blend_images.shape[0] != b_size):
        raise ValueError(f"Length of detected_centers and length of blend_list\
            must be equal to first dimension of blend_images, found \
            {len(detected_centers), len(blend_list), len(blend_images)}")
    for i in range(len(blend_list)):
        images = np.transpose(blend_images[i], axes=(2, 0, 1))
        blend_img_rgb = get_rgb_image(
            images[band_indices],
            normalize_with_image=images[band_indices])
        plt.figure(figsize=(2, 2))
        plt.imshow(blend_img_rgb)
        plt.title(f"{len(blend_list[i])} objects")
        if limits:
            plt.xlim(limits)
            plt.ylim(limits)
        plt.axis('off')
        for cent in detected_centers[i]:
            plt.plot(cent[0], cent[1], 'go', fillstyle='none')
        plt.show()
        iso_blend = isolated_images[i]
        num = iso_blend.shape[0]
        plt.figure(figsize=(2 * num, 2))
        for j in range(num):
            iso_images = np.transpose(iso_blend[j], axes=(2, 0, 1))
            iso_img_rgb = get_rgb_image(
                iso_images[band_indices],
                normalize_with_image=images[band_indices])
            plt.subplot(1, num, j + 1)
            plt.imshow(iso_img_rgb)
            if limits:
                plt.xlim(limits)
                plt.ylim(limits)
            plt.axis('off')
            if len(detected_centers[i]) > 0:
                plt.plot(detected_centers[i][j][0], detected_centers[i][j][1],
                         'go', fillstyle='none')
        plt.show()


def plot_cumulative(table, column_name, ax=None, bins=40,
                    color='red', label=None, xlabel=None):
    """Plot cumulative counts of input column_name in table.

    Args:
        table(`astropy.table.Table`) : Catalog with features as columns and
            different samples at rows.
        column_name(str): Name of column in input table who's cumulative
            counts are to be plotted.
        ax(`matplotlib.axes`, default=`None`): Matplotlib axis on which to draw
            the plot. If not provided, one is created inside.
        bins(int or sequence of scalars, optional, default=40): If bins is an
            int, it defines the number of equal-width bins in the given range
            (40, by default). If bins is a sequence, it defines a monotonically
            increasing array of bin edges, including the rightmost edge,
            allowing for non-uniform bin widths.
        color(str, default='red'): Color of cumulative counts curve.
        label(str, default=`None`): label for legend in plot.
        xlabel(str, default=`None`): x-axis label in plot. If not provided,
            then the column_name is set as x-axis label.
    """
    if xlabel is None:
        xlabel = column_name
    det_values, det_base = np.histogram(table[column_name], bins=bins)
    det_cumulative = np.cumsum(det_values)
    if label is None:
        ax.plot(det_base[:-1], det_cumulative, c=color)
    else:
        ax.plot(det_base[:-1], det_cumulative, c=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative counts')


def plot_metrics_summary(summary, num, ax=None):
    """Plot detection summary as a matrix of detection efficiency.

    Input argument num sets the maximum number of true detections for which the
    detection efficiency matrix is to be created for. Detection efficiency is
    computed for number of true objects in the range (1-num)

    Args:
        summary(`numpy.array`) : Detection summary as a table [N, 5].
        num(int): Maximum number of true objects to create matrix for. Number
            of columns in matrix will be num-1.
        ax(`matplotlib.axes`, default=`None`): Matplotlib axis on which to draw
            the plot. If not provided, one is created inside.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(wspace=0.7, hspace=0.3)
    results_table = btk.utils.get_detection_eff_matrix(summary, num)
    ax.imshow(results_table, origin='left', cmap=plt.cm.Blues)
    ax.set_xlabel("# true objects")
    # Don't print zero'th column
    ax.set_xlim([0.5, num + 0.5])
    ax.set_ylabel("# correctly detected objects")
    ax.set_xticks(np.arange(1, num + 1, 1.0))
    ax.set_yticks(np.arange(0, num + 2, 1.0))
    for (j, i), label in np.ndenumerate(results_table):
        if i == 0:
            # Don't print efficiency for zero'th column
            continue
        color = ("white" if label > 50
                 else "black" if label > 0
                 else "grey")
        ax.text(i, j, f"{label:.1f}%",
                ha='center', va='center', color=color)
        if i == j:
            rect = patches.Rectangle((i - 0.5, j - 0.5), 1, 1, linewidth=2,
                                     edgecolor='mediumpurple',
                                     facecolor='none')
            ax.add_patch(rect)
