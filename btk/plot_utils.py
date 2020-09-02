import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import btk


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
        raise ValueError(
            "Must be 3 channel in dimension 1 of image" f"Found {image.shape[0]}"
        )
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
        image (float32): Image array to convert to RGB [bands, height, width].
        normalize_with_image (float32): Image array to normalize input image
            with [bands, height, width].

    Returns:
        uint8 array [height, width, bands] of the input image.
    """
    try:
        import scarlet

        if normalize_with_image is not None:
            Q = 0.1
            minimum = np.ma.min(normalize_with_image)
            maximum = np.ma.max(normalize_with_image)
            stretch = maximum - minimum
            norm = scarlet.display.AsinhMapping(minimum=minimum, stretch=stretch, Q=Q)
        else:
            norm = None
        img_rgb = scarlet.display.img_to_rgb(image, norm=norm)

    except ImportError:
        scarlet = None
        # scarlet not installed, basic normalize image to 0-255
        if normalize_with_image is None:
            min_val = None
            max_val = None
        else:
            min_val = np.min(normalize_with_image, axis=1).min(axis=-1)
            max_val = np.max(normalize_with_image, axis=1).max(axis=-1)
        img_rgb = get_rgb(image, min_val=min_val, max_val=max_val)
    return img_rgb


def plot_blends(
    blend_images, blend_list, detected_centers=None, limits=None, band_indices=None
):
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
        band_indices (list, default=None): list of length 3 with indices of
            bands that are to be plotted in the RGB image. If pass in None,
            then default value of [1, 2, 3] is used.
    """
    if band_indices is None:
        band_indices = [1, 2, 3]
    batch_size = len(blend_list)
    if len(band_indices) != 3:
        raise ValueError(
            f"band_indices must be a list with 3 entries, not \
            {band_indices}"
        )
    if detected_centers is None:
        detected_centers = [[]] * batch_size
    if len(detected_centers) != batch_size or blend_images.shape[0] != batch_size:
        raise ValueError(
            f"Length of detected_centers and length of blend_list\
            must be equal to first dimension of blend_images, found \
            {len(detected_centers), len(blend_list), len(blend_images)}"
        )
    for i in range(batch_size):
        num = len(blend_list[i])
        images = np.transpose(blend_images[i], axes=(2, 0, 1))
        blend_img_rgb = get_rgb_image(images[band_indices])
        _, ax = plt.subplots(1, 3, figsize=(8, 3))
        ax[0].imshow(blend_img_rgb)
        if limits:
            ax[0].set_xlim(limits)
            ax[0].set_ylim(limits)
        ax[0].set_title("gri bands")
        ax[0].axis("off")
        ax[1].imshow(np.sum(blend_images[i, :, :, :], axis=2))
        ax[1].set_title("Sum")
        if limits:
            ax[1].set_xlim(limits)
            ax[1].set_ylim(limits)
        ax[1].axis("off")
        ax[2].imshow(blend_img_rgb)
        ax[2].set_title(f"{num} objects with centers")
        for entry in blend_list[i]:
            ax[2].plot(entry["dx"], entry["dy"], "rx")
        if limits:
            ax[2].set_xlim(limits)
            ax[2].set_ylim(limits)
        for cent in detected_centers[i]:
            ax[2].plot(cent[0], cent[1], "go", fillstyle="none", ms=10, mew=2)
        ax[2].axis("off")
    plt.show()


def plot_with_isolated(
    blend_images,
    isolated_images,
    blend_list,
    detected_centers=None,
    limits=None,
    band_indices=None,
):
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
        band_indices (list, default=None): list of length 3 with indices of
            bands that are to be plotted in the RGB image. If pass in None,
            then default value of [1, 2, 3] is used.
    """
    if band_indices is None:
        band_indices = [1, 2, 3]
    b_size = len(blend_list)
    if len(band_indices) != 3:
        raise ValueError(
            f"band_indices must be a list with 3 entries, not \
            {band_indices}"
        )
    if detected_centers is None:
        detected_centers = [[]] * b_size
    if (
        len(detected_centers) != b_size
        or len(isolated_images) != b_size
        or blend_images.shape[0] != b_size
    ):
        raise ValueError(
            f"Length of detected_centers and length of blend_list\
            must be equal to first dimension of blend_images, found \
            {len(detected_centers), len(blend_list), len(blend_images)}"
        )
    for i in range(len(blend_list)):
        images = np.transpose(blend_images[i], axes=(2, 0, 1))
        blend_img_rgb = get_rgb_image(
            images[band_indices], normalize_with_image=images[band_indices]
        )
        plt.figure(figsize=(2, 2))
        plt.imshow(blend_img_rgb)
        plt.title(f"{len(blend_list[i])} objects")
        if limits:
            plt.xlim(limits)
            plt.ylim(limits)
        plt.axis("off")
        for cent in detected_centers[i]:
            plt.plot(cent[0], cent[1], "go", fillstyle="none")
        plt.show()
        iso_blend = isolated_images[i]
        num = iso_blend.shape[0]
        plt.figure(figsize=(2 * num, 2))
        for j in range(num):
            iso_images = np.transpose(iso_blend[j], axes=(2, 0, 1))
            iso_img_rgb = get_rgb_image(
                iso_images[band_indices], normalize_with_image=images[band_indices]
            )
            plt.subplot(1, num, j + 1)
            plt.imshow(iso_img_rgb)
            if limits:
                plt.xlim(limits)
                plt.ylim(limits)
            plt.axis("off")
            if len(detected_centers[i]) > 0:
                plt.plot(
                    detected_centers[i][j][0],
                    detected_centers[i][j][1],
                    "go",
                    fillstyle="none",
                )
        plt.show()


def plot_cumulative(
    table, column_name, ax=None, bins=40, color="red", label=None, xlabel=None
):
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
    ax.set_ylabel("Cumulative counts")


def plot_metrics_summary(summary, num, ax=None, wspace=0.2, skip_zero=True):
    """Plot detection summary as a matrix of detection efficiency.

    Input argument num defines the maximum number of true objects per blend in
    the defined test set for which the detection efficiency matrix is to be
    computed. Detection efficiency matrix is plotted for columns 1 - num true
    objects per blend, unless skip_zero is set to False, in which case column
    for 0 true objects is also displayed.

    Args:
        summary (`numpy.array`): Detection summary as a table [N, 5].
        num (int): Maximum number of true objects to plot matrix for. Number
            of columns in matrix will be num-1 if skip_zero is True.
        ax(`matplotlib.axes`, default=`None`): Matplotlib axis on which to draw
            the plot. If not provided, one is created inside.
        wspace (float): Amount of width reserved for space between subplots,
            expressed as a fraction of the average axis width.
        skip_zero (bool): If True, then column corresponding to 0 true objects
            is not shown (default is True).
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(wspace=wspace)
    results_table = btk.utils.get_detection_eff_matrix(summary, num)
    ax.imshow(results_table, origin="left", cmap=plt.cm.Blues)
    ax.set_xlabel("# true objects")
    if skip_zero:
        # Don't print zero'th column
        ax.set_xlim([0.5, num + 0.5])
        ax.set_xticks(np.arange(1, num + 1, 1.0))
    else:
        ax.set_xlim([-0.5, num + 0.5])
        ax.set_xticks(np.arange(0, num + 1, 1.0))
    ax.set_ylabel("# correctly detected objects")
    ax.set_yticks(np.arange(0, num + 2, 1.0))
    for (j, i), label in np.ndenumerate(results_table):
        if skip_zero and i == 0:
            # Don't print efficiency for zero'th column
            continue
        color = "white" if label > 50 else "black" if label > 0 else "grey"
        ax.text(i, j, f"{label:.1f}%", ha="center", va="center", color=color)
        if i == j:
            rect = patches.Rectangle(
                (i - 0.5, j - 0.5),
                1,
                1,
                linewidth=2,
                edgecolor="mediumpurple",
                facecolor="none",
            )
            ax.add_patch(rect)


def show_scarlet_residual(blend, observation, limits=(30, 90)):
    """Plot scarlet model and residual image in rgb and i band.

    Note: this requires scarlet to be installed.
    Args:
        blend: output of scarlet containing blend fit.
        observation: `~scarlet.Observation`
        limits(list, default=`None`): List of start and end coordinates to
        display image within. Note: limits are applied to both height and
        width dimensions.
    """
    try:
        import scarlet
        import scarlet.display

        figsize1 = (12, 12)
        figsize2 = (16, 16)
        fig, ax = plt.subplots(1, 4, figsize=figsize1)
        fig2, ax2 = plt.subplots(1, 4, figsize=figsize2)
        model = blend.get_model()
        ax[0].imshow(scarlet.display.img_to_rgb(model))
        ax[0].set_title("Model")
        cbar = ax2[0].imshow(model[4] / 10 ** 3)
        divider1 = make_axes_locatable(ax2[0])
        cax = divider1.append_axes("right", size="4%", pad=0.05)
        clb = plt.colorbar(cbar, cax=cax)
        clb.ax.set_title("$10^3$", size=8)
        model = observation.render(model)
        ax[1].imshow(scarlet.display.img_to_rgb(model))
        ax[1].set_title("Model Rendered")
        cbar = ax2[1].imshow(model[4] / 10 ** 3)
        divider1 = make_axes_locatable(ax2[1])
        cax = divider1.append_axes("right", size="4%", pad=0.05)
        clb = plt.colorbar(cbar, cax=cax)
        clb.ax.set_title("$10^3$", size=8)
        ax[2].imshow(scarlet.display.img_to_rgb(observation.images))
        ax[2].set_title("Observation")
        cbar = ax2[2].imshow(observation.images[4] / 10 ** 3)
        divider1 = make_axes_locatable(ax2[2])
        cax = divider1.append_axes("right", size="4%", pad=0.05)
        clb = plt.colorbar(cbar, cax=cax)
        clb.ax.set_title("$10^3$", size=8)
        residual = observation.images - model
        ax[3].imshow(scarlet.display.img_to_rgb(residual))
        ax[3].set_title("Residual")
        cbar = ax2[3].imshow(residual[4] / 10 ** 3)
        divider1 = make_axes_locatable(ax2[3])
        cax = divider1.append_axes("right", size="4%", pad=0.05)
        clb = plt.colorbar(cbar, cax=cax)
        clb.ax.set_title("$10^3$", size=8)
        fig.tight_layout()
        for a in ax:
            a.set_xlim(limits)
            a.set_ylim(limits)
        for a in ax2:
            a.axis("off")
            a.set_xlim(limits)
            a.set_ylim(limits)
        plt.show()

    except ImportError:
        print("Scarlet is needed to use this function.")
