"""Utility functions for plotting and displaying images in BTK."""
import os

import ipywidgets as widgets
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.visualization import make_lupton_rgb
from IPython.display import clear_output, display
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_quadrant(center, image_size):
    """Get the correct quadrant coordinates for plotting an inset plot.

    Args:
        center (tuple) : Coordinates the inset plot is centered on.
        image_size (int): Size of the image.

    Returns:
        The coordinates to pass to matplotlib.
    """
    if center[0] >= image_size // 2 and center[1] >= image_size // 2:
        return [0.03, 0.6, 0.37, 0.37]
    elif center[0] >= image_size // 2 and center[1] <= image_size // 2:
        return [0.03, 0.03, 0.37, 0.37]
    elif center[0] <= image_size // 2 and center[1] >= image_size // 2:
        return [0.6, 0.6, 0.37, 0.37]
    else:
        return [0.6, 0.03, 0.37, 0.37]


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


def get_rgb_image(image, norm="linear", Q=0.1):
    """Returns RGB (0-255) image corresponding to the input 3 band image.

    If scarlet.display is imported then the normalization is performed by
    scarlet Asinh function. If not, a basic normalization is performed.

    Args:
        image (array_like): Image array to convert to RGB [bands, height, width].
        norm (str): Stretch to apply to the images. Must be one of "linear" or "asinh".
        Q (float): Smoothing parameter for the "asinh" norm.

    Returns:
        uint8 array [height, width, bands] of the input image.
    """
    if norm == "asinh":
        img_rgb = make_lupton_rgb(
            image[0],
            image[1],
            image[2],
            stretch=np.max(image) - np.min(image),
            Q=Q,
            minimum=np.min(image),
        )
    else:
        img_rgb = get_rgb(image)
    return img_rgb


def get_image(image, bands, rgb=False, norm="linear"):
    """Returns the rgb image if rgb is true, or a monochromatic image if it is false.

    Args:
        image (numpy Array): Contains the image [bands,height,width]
        bands (list) : list of the bands to be used. Should be of length 3
            for RGB images and 1 for monochromatic images.
        rgb (bool) : indicates if the returned image should be RGB or
            monochromatic.
        norm (str): Stretch to apply to the RGB images. Must be one of "linear" or "asinh".

    Returns:
        The requested image.
    """
    if rgb:
        return get_rgb_image(image[bands], norm=norm)
    else:
        return image[bands[0]]


def plot_blends(
    blend_images,
    blend_list,
    detected_centers=None,
    limits=None,
    band_indices=None,
    norm="linear",
    Q=0.1,
):
    """Plots blend images as RGB image, sum in all bands, and RGB image with centers of objects.

    Outputs of btk draw are plotted here. Blend_list must contain true  centers
    of the objects. If detected_centers are input, then the centers are also
    shown in the third panel along with the true centers.

    Args:
        blend_images (array_like): Array of blend scene images to plot
            [batch, bands, height, width].
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
            then default value of [3, 2, 1] is used.
        norm (str): Stretch to apply to the images. Must be one of "linear" or "asinh".
        Q (float): Smoothing parameter for the "asinh" norm.
    """
    if band_indices is None:
        band_indices = [3, 2, 1]
    batch_size = len(blend_list)
    if len(band_indices) != 3:
        raise ValueError(f"band_indices must be a list with 3 entries, not {band_indices}")
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
        images = blend_images[i]
        blend_img_rgb = get_rgb_image(images[band_indices], norm=norm, Q=Q)
        _, ax = plt.subplots(1, 3, figsize=(20, 10))
        ax[0].imshow(blend_img_rgb)
        if limits:
            ax[0].set_xlim(limits)
            ax[0].set_ylim(limits)
        ax[0].set_title("gri bands")
        ax[0].axis("off")
        ax[1].imshow(np.sum(blend_images[i, :, :, :], axis=0))
        ax[1].set_title("Sum over bands")
        if limits:
            ax[1].set_xlim(limits)
            ax[1].set_ylim(limits)
        ax[1].axis("off")
        ax[2].imshow(blend_img_rgb)
        ax[2].set_title(f"{num} objects with centers")
        for entry in blend_list[i]:
            ax[2].plot(entry["x_peak"], entry["y_peak"], "rx")
        if limits:
            ax[2].set_xlim(limits)
            ax[2].set_ylim(limits)
        for cent in detected_centers[i]:
            ax[2].plot(cent[0], cent[1], "go", fillstyle="none", ms=10, mew=2)
        ax[2].axis("off")
    plt.show()


def plot_with_isolated(
    blend_images, isolated_images, blend_list, limits=None, band_indices=None, norm="linear"
):
    """Plots blend images and isolated images of all objects in the blend as RGB images.

    Outputs of btk draw are plotted here. Blend_list must contain true  centers
    of the objects.

    Args:
        blend_images(array_like): Array of blend scene images to plot
            [batch, height, width, bands].
        isolated_images (array_like): Array of isolated object images to plot
            [batch, max number of objects, height, width, bands].
        blend_list(list): List of `astropy.table.Table` with entries of true
            objects. Length of list must be the batch size.
        limits(list, default=`None`): List of start and end coordinates to
            display image within. Note: limits are applied to both height and
            width dimensions.
        band_indices (list, default=None): list of length 3 with indices of
            bands that are to be plotted in the RGB image. If pass in None,
            then default value of [3, 2, 1] is used.
        norm (str): Stretch to apply to the images. Must be one of "linear" or "asinh".
    """
    if band_indices is None:
        band_indices = [3, 2, 1]
    if len(band_indices) not in [1, 3]:
        raise ValueError(
            f"band_indices must be a list with 1 or 3 entries, not \
            {band_indices}"
        )
    rgb = len(band_indices) == 3
    for i in range(len(blend_list)):
        images = blend_images[i]
        blend_img = get_rgb_image(images[band_indices], norm=norm) if rgb else images[band_indices]
        plt.figure(figsize=(2, 2))
        plt.imshow(blend_img)
        plt.title(f"{len(blend_list[i])} objects")
        if limits:
            plt.xlim(limits)
            plt.ylim(limits)
        plt.axis("off")
        plt.show()
        iso_blend = isolated_images[i]
        num = iso_blend.shape[0]
        plt.figure(figsize=(2 * num, 2))
        for j in range(num):
            iso_images = iso_blend[j]
            iso_img = (
                get_rgb_image(iso_images[band_indices], norm=norm)
                if rgb
                else iso_images[band_indices]
            )
            plt.subplot(1, num, j + 1)
            plt.imshow(iso_img)
            if limits:
                plt.xlim(limits)
                plt.ylim(limits)
            plt.axis("off")
        plt.show()


def plot_with_deblended(
    blend_images,
    isolated_images,
    blend_list,
    detection_catalogs,
    deblended_images,
    matches,
    indexes=[0],
    band_indices=[3, 2, 1],
    norm="linear",
    noise_level=1.0,
):
    """Plots blend images, along with isolated, deblended and residual images of objects in a blend.

    Outputs of btk draw are plotted here. blend_images, isolated_images and blend_list
    are expected to be the corresponding entries from the output of a DrawBlendsGenerator,
    detection_catalogs and deblended_images are taken from the output of a MeasureGenerator,
    and matches is the matches entry of a MetricsGenerator (or the compute_metrics function).

    Args:
        blend_images(array_like): Array of blend scene images to plot
            [batch, height, width, bands].
        isolated_images (array_like): Array of isolated object images to plot
            [batch, max number of objects, bands, height, width].
        blend_list (list): List of `astropy.table.Table` with entries of true
            objects. Length of list must be the batch size.
        detection_catalogs (list): List of `astropy.table.Table` with entries of
            detected objects.
        deblended_images (list): List of arrays containing the deblended images with
            length 'batch' and array shape [bands,height,width]
        matches (list): List of `astropy.table.Table` with entries corresponding to the
            true galaxies. The column 'match_detected_id' must contain the id of the
            matching true galaxy.
        indexes (list): List of the indexes of the blends you want to plot.
        band_indices (list): List of the bands to plot. Should have either 3 elements
            for RGB images, or 1 for monochromatic images.
        norm (str): Stretch to apply to the RGB images. Must be one of "linear" or "asinh".
        noise_level (float) : Normalization to apply to monochromatic images. Should be the
            standard deviation of the noise.

    """
    sns.set_context("notebook")
    if len(band_indices) not in [1, 3]:
        raise ValueError(
            f"band_indices must be a list with 1 or 3 entries, not \
            {band_indices}"
        )
    rgb = len(band_indices) == 3
    for i in indexes:
        nrow = len(matches[i])
        fig = plt.figure(constrained_layout=True, figsize=(10, 10 + 5 * nrow))
        spec = fig.add_gridspec(
            nrow + 1, 3, height_ratios=[nrow] + [1] * nrow, width_ratios=[1, 1, 1]
        )
        ax = [[] for j in range(nrow + 1)]
        ax[0].append(fig.add_subplot(spec[0, :]))
        ax[0][0].imshow(
            get_rgb_image(blend_images[i][band_indices], norm=norm)
            if rgb
            else blend_images[i][band_indices[0]]
        )
        ax[0][0].scatter(
            blend_list[i]["x_peak"],
            blend_list[i]["y_peak"],
            color="red",
            marker="x",
            label="True centroids",
            s=150,
            linewidth=3,
        )
        ax[0][0].scatter(
            detection_catalogs[i]["x_peak"],
            detection_catalogs[i]["y_peak"],
            color="blue",
            marker="+",
            label="Detected centroids",
            s=150,
            linewidth=3,
        )
        ax[0][0].set_title("Blended image", fontsize=18)
        ax[0][0].legend(fontsize=16)
        for k in range(nrow):
            match = matches[i]["match_detected_id"][k]
            if not rgb:
                vmin = np.min(
                    np.minimum.reduce(
                        [
                            isolated_images[i][k],
                            deblended_images[i][match],
                            isolated_images[i][k] - deblended_images[i][match],
                        ]
                    )
                )
                vmax = np.max(
                    np.maximum.reduce(
                        [
                            isolated_images[i][k],
                            deblended_images[i][match],
                            isolated_images[i][k] - deblended_images[i][match],
                        ]
                    )
                )
            else:
                vmin, vmax = None, None
            ax[k].append(fig.add_subplot(spec[k + 1, 0]))
            ax[k][-1].imshow(
                get_image(isolated_images[i][k], band_indices, rgb, norm=norm), vmin=vmin, vmax=vmax
            )
            if k == 0:
                ax[k][-1].set_title("True galaxies")
            if match != -1:
                ax[k].append(fig.add_subplot(spec[k + 1, 1]))
                ax[k][-1].imshow(
                    get_image(deblended_images[i][match], band_indices, rgb, norm=norm),
                    vmin=vmin,
                    vmax=vmax,
                )
                if k == 0:
                    ax[k][-1].set_title("Detected galaxies")
                if not rgb:
                    ax[k][-1].set_yticks([])
                ax[k].append(fig.add_subplot(spec[k + 1, 2]))
                ax[k][-1].imshow(
                    get_image(
                        isolated_images[i][k] - deblended_images[i][match],
                        band_indices,
                        rgb,
                        norm=norm,
                    ),
                    vmin=vmin,
                    vmax=vmax,
                )
                if k == 0:
                    ax[k][-1].set_title("Residuals")
                if not rgb:
                    ax[k][-1].set_yticks([])
                    divider = make_axes_locatable(ax[k][-1])
                    ax[k].append(divider.append_axes("right", size="5%", pad=0.05))
                    normalize = Normalize(vmin=vmin / noise_level, vmax=vmax / noise_level)
                    cbar = fig.colorbar(cm.ScalarMappable(normalize), cax=ax[k][-1])
                    cbar.set_label("SNR")
                # inset axes....
                axins = ax[k][-2].inset_axes(
                    get_quadrant(
                        [blend_list[i]["x_peak"][k], blend_list[i]["y_peak"][k]],
                        isolated_images[i][k].shape[-1],
                    )
                )
                axins.imshow(
                    get_image(
                        isolated_images[i][k] - deblended_images[i][match],
                        band_indices,
                        rgb,
                        norm=norm,
                    ),
                    origin="lower",
                )
                x1, x2, y1, y2 = (
                    blend_list[i]["x_peak"][k] - 12,
                    blend_list[i]["x_peak"][k] + 12,
                    blend_list[i]["y_peak"][k] - 12,
                    blend_list[i]["y_peak"][k] + 12,
                )
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticks([])
                axins.set_yticks([])

                ax[k][-2].indicate_inset_zoom(axins, edgecolor="black")
        spec.tight_layout(fig)
        plt.show()


def plot_efficiency_matrix(eff_matrix, ax=None, wspace=0.2, skip_zero=True):
    """Plot detection summary as a matrix of detection efficiency.

    Args:
        eff_matrix (`numpy.array`): Efficiency matrix
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
    results_table = eff_matrix
    num = eff_matrix.shape[0] - 2
    ax.imshow(results_table, origin="lower", cmap=plt.cm.Blues)
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


def plot_metrics_distribution(metric_array, metric_name, ax=None, bins=50, upper_quantile=1.0):
    """Plot an histogram of the distribution with mean and median.

    Args:
        metric_array: Contains the data
        metric_name (str): name(s) of the metric(s)
        ax (matplotlib.axes.Axes): ax on which the plot should be drawn
        bins (int): Optional argument for the number of bins.
        upper_quantile (float): Quantile from which to cut
    """
    ax = plt.gca() if ax is None else ax
    quantile = np.quantile(metric_array, upper_quantile)
    m_filtered = metric_array[metric_array <= quantile]

    sns.histplot(x=m_filtered, ax=ax)
    ax.set_xlabel(metric_name)
    mean = np.mean(m_filtered)
    ax.axvline(mean, linestyle="--", color="blue", label="Mean")
    median = np.median(m_filtered)
    ax.axvline(median, linestyle="--", color="red", label="Median")
    ax.legend()


def plot_metrics_correlation(
    metric_x, metric_y, metric_x_name, metric_y_name, ax=None, upper_quantile=1.0, style="scatter"
):
    """Plot a scatter plot between two quantities.

    Args:
        metric_x: Contains the data for the x axis
        metric_y:Contains the data for the y axis
        metric_x_name (str): name of the x metric
        metric_y_name (str): name of the y metric
        ax (matplotlib.axes.Axes): ax on which the plot should be drawn
        upper_quantile (float): Quantile from which to cut
        style (str): Style of the plot, can be "scatter" or "heatmap"

    """
    ax = plt.gca() if ax is None else ax
    quantile = np.quantile(metric_y, upper_quantile)
    metric_x = metric_x[metric_y < quantile]
    metric_y = metric_y[metric_y < quantile]
    if style == "heatmap":
        heatmap, xedges, yedges = np.histogram2d(metric_x, metric_y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(
            heatmap.T, extent=extent, origin="lower", aspect="auto", label=metric_y_name, cmap="hot"
        )
    elif style == "truth":
        heatmap, xedges, yedges = np.histogram2d(metric_x, metric_y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(heatmap.T, extent=extent, origin="lower", label=metric_y_name, cmap="Greens")
        x = np.linspace(np.min(metric_x), np.max(metric_x), 10)
        ax.plot(x, x, linestyle="--", color="black")
    elif style == "scatter":
        ax.scatter(metric_x, metric_y, label=metric_y_name)
    else:
        raise ValueError("Invalid style")
    ax.set_xlabel(metric_x_name)
    ax.set_ylabel(metric_y_name)


def plot_gal_parameters(blend_list, context="talk"):
    """Plots histograms for the magnitude and the size of the galaxies in a batch.

    Args:
        blend_list (list): List of astropy Table. Should be obtained from the output of a
                            DrawBlendsGenerator.
        context (str): Context for seaborn; see seaborn documentation for details.
                        Can be one of "paper", "notebook", "talk", and "poster".
    """
    sns.set_context(context)
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    plot_metrics_distribution(
        np.concatenate([blend_list[i]["ref_mag"] for i in range(len(blend_list))]),
        "Magnitude",
        ax[0],
    )
    plot_metrics_distribution(
        np.concatenate([blend_list[i]["btk_size"] for i in range(len(blend_list))]),
        "Size (in pixels)",
        ax[1],
    )
    plt.show()


def plot_metrics_summary(  # noqa: C901
    metrics_results,
    target_meas_keys=[],
    target_meas_limits=[],
    n_bins_target=30,
    aliases={},
    save_path=None,
    context="talk",
    interactive=False,
):
    """Plot metrics directly from the MetricsGenerator output.

    Args:
        metrics_results (dict): Output of a MetricsGenerator.
        target_meas_keys (list): List of the keys for the target measures.
        target_meas_limits (list): List of tuples indicating the limits for the plots
                                   of the target measures
        n_bins_target (int): Number of bins for the target measure plots
        aliases (dict) : Replaces the names contained in the keys by their
                                associated values. Used to get proper names in the
                                figures.
        save_path (str): Path to the folder where the figures should be saved.
        context (str): Context for seaborn; see seaborn documentation for details.
                        Can be one of "paper", "notebook", "talk", and "poster".
        interactive (bool): Specifies if the plot should be interactive.

    """
    sns.set_context(context)
    # Keys corresponding to the measure functions
    measure_keys = list(metrics_results["galaxy_summary"].keys())

    # We need to handle the multiresolution case
    if isinstance(metrics_results["galaxy_summary"][measure_keys[0]], dict):
        survey_keys = list(metrics_results["galaxy_summary"][measure_keys[0]].keys())
        gal_summary_keys = list(
            metrics_results["galaxy_summary"][measure_keys[0]][survey_keys[0]].keys()
        )
        multiresolution = True
        # Limits for widgets
        min_mag = np.min(
            metrics_results["galaxy_summary"][measure_keys[0]][survey_keys[0]]["ref_mag"]
        )
        max_mag = np.max(
            metrics_results["galaxy_summary"][measure_keys[0]][survey_keys[0]]["ref_mag"]
        )
        min_size = np.min(
            metrics_results["galaxy_summary"][measure_keys[0]][survey_keys[0]]["btk_size"]
        )
        max_size = np.max(
            metrics_results["galaxy_summary"][measure_keys[0]][survey_keys[0]]["btk_size"]
        )
    else:
        gal_summary_keys = list(metrics_results["galaxy_summary"][measure_keys[0]].keys())
        multiresolution = False
        min_mag = np.min(metrics_results["galaxy_summary"][measure_keys[0]]["ref_mag"])
        max_mag = np.max(metrics_results["galaxy_summary"][measure_keys[0]]["ref_mag"])
        min_size = np.min(metrics_results["galaxy_summary"][measure_keys[0]]["btk_size"])
        max_size = np.max(metrics_results["galaxy_summary"][measure_keys[0]]["btk_size"])
    plot_keys = ["reconstruction", "segmentation", "eff_matrix"] + target_meas_keys + ["custom"]

    if interactive:
        layout = widgets.Layout(width="auto")
        # Checkboxes for selecting the measure function
        measure_functions_dict = {
            key: widgets.Checkbox(description=key, value=False, layout=layout)
            for key in measure_keys
        }
        measure_functions = [measure_functions_dict[key] for key in measure_keys]
        measure_functions_widget = widgets.VBox(measure_functions)
        measure_title_widget = widgets.HTML("<em>Measure functions</em>")
        measure_vbox = widgets.VBox([measure_title_widget, measure_functions_widget])
        # Checkboxes for selecting the survey (if multiresolution)
        if multiresolution:
            surveys_dict = {
                key: widgets.Checkbox(description=key, value=False, layout=layout)
                for key in survey_keys
            }
            surveys = [surveys_dict[key] for key in survey_keys]
            surveys_widget = widgets.VBox(surveys)
            surveys_title_widget = widgets.HTML("<em>Surveys</em>")
            surveys_vbox = widgets.VBox([surveys_title_widget, surveys_widget])
            measure_surveys_widget = widgets.VBox([measure_vbox, surveys_vbox])
        else:
            measure_surveys_widget = measure_vbox
        # Sliders to filter based on parameters
        blendedness_widget = widgets.FloatRangeSlider(
            description="Blendedness",
            value=[0, 1.0],
            min=0,
            max=1.0,
            step=0.01,
            continuous_update=False,
        )
        magnitude_widget = widgets.FloatRangeSlider(
            description="Magnitude",
            value=[min_mag, max_mag],
            min=min_mag,
            max=max_mag,
            step=0.01,
            continuous_update=False,
        )
        size_widget = widgets.FloatRangeSlider(
            description="Size",
            value=[min_size, max_size],
            min=min_size,
            max=max_size,
            step=0.01,
            continuous_update=False,
        )
        filter_vbox = widgets.VBox([blendedness_widget, magnitude_widget, size_widget])
        # Checkboxes for selecting which metrics will be plotted
        plot_selection_dict = {
            key: widgets.Checkbox(description=key, value=False) for key in plot_keys
        }
        plot_selection = [plot_selection_dict[key] for key in plot_keys]
        plot_selection_widget = widgets.VBox(plot_selection)
        # Dropdowns for selecting the parameters for the custom plot
        custom_x_widget_drop = widgets.Dropdown(
            options=gal_summary_keys,
            description="X coordinate value",
            layout=layout,
        )
        custom_y_widget_drop = widgets.Dropdown(
            options=gal_summary_keys,
            description="Y coordinate value",
            layout=layout,
        )
        custom_x_widget_log = widgets.Checkbox(description="Log scale", value=False, layout=layout)
        custom_x_widget = widgets.HBox([custom_x_widget_drop, custom_x_widget_log])
        custom_y_widget_log = widgets.Checkbox(description="Log scale", value=False, layout=layout)
        custom_y_widget = widgets.HBox([custom_y_widget_drop, custom_y_widget_log])

        plot_selection_vbox = widgets.VBox(plot_selection + [custom_x_widget, custom_y_widget])

        hbox = widgets.HBox([measure_surveys_widget, filter_vbox, plot_selection_vbox])
        display(hbox)

    # This function is called everytime the values of the widget change, and at the start
    def draw_plots(value):
        # If there are no widgets we use default values, else we get all the values
        if interactive:
            clear_output()
            display(hbox)
            meas_func_names = [w.description for w in measure_functions_widget.children if w.value]
            if multiresolution:
                surveys = [w.description for w in surveys_widget.children if w.value]
            blendedness_limits = blendedness_widget.value
            mag_limits = magnitude_widget.value
            size_limits = size_widget.value
            custom_x = custom_x_widget_drop.value
            custom_y = custom_y_widget_drop.value
            custom_x_log = custom_x_widget_log.value
            custom_y_log = custom_y_widget_log.value
            plot_selections = {w.description: w.value for w in plot_selection_widget.children}
        else:
            meas_func_names = measure_keys
            if multiresolution:
                surveys = survey_keys
            blendedness_limits = [0, 1]
            mag_limits = [min_mag, max_mag]
            size_limits = [min_size, max_size]
            plot_selections = {w: True for w in plot_keys}
            plot_selections["custom"] = False

        # If no measure function (or no surveys if multiresolution) is ticked, plot nothing
        if len(meas_func_names) == 0:
            return 0
        if multiresolution and len(surveys) == 0:
            return 0

        # Group all the data into a dataframe for using seaborn
        if multiresolution:
            dataframes = {}
            couples = []
            for f_name in meas_func_names:
                for s_name in surveys:
                    couples.append(f_name + "_" + s_name)
                    dataframes[f_name + "_" + s_name] = metrics_results["galaxy_summary"][f_name][
                        s_name
                    ].to_pandas()
            concatenated = pd.concat(
                [dataframes[c].assign(measure_function=c) for c in couples], ignore_index=True
            )
        else:
            dataframes = {}
            for f_name in meas_func_names:
                dataframes[f_name] = metrics_results["galaxy_summary"][f_name].to_pandas()
            concatenated = pd.concat(
                [dataframes[f_name].assign(measure_function=f_name) for f_name in meas_func_names],
                ignore_index=True,
            )
        concatenated.replace(aliases, inplace=True)
        concatenated.rename(columns={"measure_function": "Measure function"}, inplace=True)
        # Filter the data for the different parameters
        concatenated = concatenated.loc[
            (concatenated["blendedness"] >= blendedness_limits[0])
            & (concatenated["blendedness"] <= blendedness_limits[1])
        ]
        concatenated = concatenated.loc[
            (concatenated["ref_mag"] >= mag_limits[0]) & (concatenated["ref_mag"] <= mag_limits[1])
        ]
        concatenated = concatenated.loc[
            (concatenated["btk_size"] >= size_limits[0])
            & (concatenated["btk_size"] <= size_limits[1])
        ]
        for k in target_meas_keys:
            concatenated["delta_" + k] = concatenated[k] - concatenated[k + "_true"]

        # Custom scatter plot for the two chosen quantities
        if plot_selections["custom"]:
            fig, ax = plt.subplots(figsize=(15, 15))
            sns.scatterplot(
                data=concatenated, x=custom_x, y=custom_y, hue="Measure function", ax=ax
            )
            if custom_x_log:
                ax.set_xscale("log")
            if custom_y_log:
                ax.set_yscale("log")
            plt.show()

        # Histograms for the reconstruction metrics
        if "msr" in concatenated and plot_selections["reconstruction"]:
            fig, ax = plt.subplots(3, 1, figsize=(20, 30))
            fig.suptitle("Distribution of reconstruction metrics", fontsize=48)
            sns.histplot(
                concatenated, x="msr", hue="Measure function", bins=30, ax=ax[0], log_scale=True
            )
            ax[0].set_xlabel("Mean square residual")
            sns.histplot(concatenated, x="psnr", hue="Measure function", bins=30, ax=ax[1])
            ax[1].set_xlabel("Peak Signal-to-Noise Ratio")
            sns.histplot(concatenated, x="ssim", hue="Measure function", bins=30, ax=ax[2])
            ax[2].set_xlabel("Structure Similarity Index")
            if save_path is not None:
                plt.savefig(os.path.join(save_path, "distributions_reconstruction.png"))
            plt.show()

        # Histograms for the segmentation metrics
        if "iou" in concatenated and plot_selections["segmentation"]:
            fig, ax = plt.subplots(figsize=(20, 10))
            fig.suptitle("Distribution of segmentation metrics", fontsize=48)
            sns.histplot(concatenated, x="iou", hue="Measure function", ax=ax, bins=30)
            ax.set_xlabel("Intersection-over-Union")
            if save_path is not None:
                plt.savefig(os.path.join(save_path, "distributions_segmentation.png"))
            plt.show()

        # Plots for the measure functions
        selected_target_meas = [m for m in target_meas_keys if plot_selections[m]]
        if selected_target_meas != []:
            n_target_meas = len(selected_target_meas)
            height_ratios = list(np.concatenate([[3, 1] for i in range(n_target_meas)]))
            fig, ax = plt.subplots(
                2 * n_target_meas,
                1,
                figsize=(10, 13.33 * n_target_meas),
                gridspec_kw={"height_ratios": height_ratios},
            )
            fig.suptitle("Target measures", fontsize=48)

            for i, k in enumerate(selected_target_meas):
                sns.scatterplot(
                    data=concatenated,
                    x=k,
                    y=k + "_true",
                    hue="Measure function",
                    ax=ax[2 * i],
                    marker="o",
                    alpha=0.7,
                )
                ax[2 * i].set(
                    xlabel="Measured " + k,
                    ylabel="True " + k,
                    xlim=target_meas_limits[i],
                    ylim=target_meas_limits[i],
                )
                xlow, xhigh = ax[2 * i].get_xlim()
                x = np.linspace(xlow, xhigh, 10)
                ax[2 * i].plot(x, x, linestyle="--", color="black", zorder=-10)

                mag_low = np.min(concatenated["ref_mag"])
                mag_high = np.max(concatenated["ref_mag"])
                for meas_func in measure_keys:
                    bins = np.linspace(mag_low, mag_high, n_bins_target)
                    labels = np.digitize(concatenated["ref_mag"], bins)
                    means = []
                    stds = []
                    to_delete = []
                    for j in range(1, n_bins_target):
                        mean = np.mean(
                            concatenated["delta_" + k][
                                (labels == j) & (concatenated["Measure function"] == meas_func)
                            ]
                        )
                        if not np.isnan(mean):
                            means.append(mean)
                            stds.append(
                                np.std(
                                    concatenated["delta_" + k][
                                        (labels == j)
                                        & (concatenated["Measure function"] == meas_func)
                                    ]
                                )
                            )
                        else:
                            to_delete.append(j)
                    bins = np.delete(bins, to_delete)
                    ax[2 * i + 1].errorbar(
                        bins[1:] - (mag_high - mag_low) / n_bins_target, means, stds
                    )

                ax[2 * i + 1].plot(
                    np.linspace(mag_low, mag_high, 10),
                    np.zeros((10)),
                    linestyle="--",
                    color="black",
                    zorder=-10,
                )
                ax[2 * i + 1].set_xlabel("Magnitude")  # noqa: W605
                ax[2 * i + 1].set_ylabel(f"$\\Delta${k}")  # noqa: W605
            plt.tight_layout()

            if save_path is not None:
                plt.savefig(os.path.join(save_path, "scatter_target_measures.png"))
            plt.show()

        # Plotting the efficiency matrices
        if plot_selections["eff_matrix"]:
            fig, ax = plt.subplots(1, len(meas_func_names), figsize=(15 * len(meas_func_names), 15))
            fig.suptitle("Efficiency matrices", fontsize=48)
            if len(meas_func_names) == 1:
                ax = [ax]
            for i, k in enumerate(meas_func_names):
                if multiresolution:
                    plot_efficiency_matrix(
                        metrics_results["detection"][k][survey_keys[0]]["eff_matrix"], ax=ax[i]
                    )
                else:
                    plot_efficiency_matrix(metrics_results["detection"][k]["eff_matrix"], ax=ax[i])
                ax[i].set_title(k)
            if save_path is not None:
                plt.savefig(os.path.join(save_path, "efficiency_matrices.png"))
            plt.show()

    # Set the widgets to update the plots if modified
    if interactive:
        blendedness_widget.observe(draw_plots, "value")
        magnitude_widget.observe(draw_plots, "value")
        size_widget.observe(draw_plots, "value")
        for k in measure_keys:
            measure_functions_dict[k].observe(draw_plots, "value")
        if multiresolution:
            for k in survey_keys:
                surveys_dict[k].observe(draw_plots, "value")
        for k in plot_keys:
            plot_selection_dict[k].observe(draw_plots, "value")
        custom_x_widget_drop.observe(draw_plots, "value")
        custom_y_widget_drop.observe(draw_plots, "value")
        custom_x_widget_log.observe(draw_plots, "value")
        custom_y_widget_log.observe(draw_plots, "value")
    else:
        draw_plots(None)
