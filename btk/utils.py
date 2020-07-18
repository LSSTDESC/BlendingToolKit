"""Contains functions to perform detection, deblending and measurement
    on images.
"""

from functools import partial

import astropy.table
import numpy as np
import skimage.feature

import btk.create_blend_generator
from btk import plot_utils
from btk.compute_metrics import Metrics_params
from btk.measure import Measurement_params


class SEP_params(Measurement_params):
    """Class to perform detection and deblending with SEP"""

    def __init__(self):
        self.catalog = None
        self.segmentation = None

    def get_centers(self, image):
        """Return centers detected when object detection and photometry
        is done on input image with SEP.

        It also initializes the self.catalog and self.segmentation attributes
        of the class object.
        Args:
            image: Image (single band) of galaxy to perform measurement on.
        Returns:
            centers: x and y coordinates of detected  centroids

        """
        sep = __import__('sep')
        bkg = sep.Background(image)
        self.catalog, self.segmentation = sep.extract(
            image, 1.5, err=bkg.globalrms, segmentation_map=True)
        centers = np.stack((self.catalog['x'], self.catalog['y']), axis=1)
        return centers

    def get_deblended_images(self, data, index):
        """Performs SEP detection on the band-coadd image and returns the
        detected peaks.

        Args:
            data (dict): Output generated by btk.draw_blends containing blended
                images, isolated images, observing conditions and blend
                catalog, for a given batch.
            index (int): Index number of blend scene in the batch to preform
                measurement on.

        Returns:
            dict with the centers of sources detected by SEP detection
            algorithm.
        """
        image = np.mean(data['blend_images'][index], axis=2)
        peaks = self.get_centers(image)
        return {'deblend_image': None, 'peaks': peaks}


def get_psf_sky(obs_cond, psf_stamp_size):
    """Returns postage stamp image of the PSF and mean background sky
    level value saved in the input obs_cond class
    Args:
        obs_cond:`descwl.survey.Survey` class describing observing
            conditions.
        psf_stamp_size: Size of postage stamp to draw PSF on in pixels.
    Returns:
        `np.ndarray`: Postage stamp image of PSF
        float: Mean of sky background

    """
    mean_sky_level = obs_cond.mean_sky_level
    psf = obs_cond.psf_model
    psf_image = psf.drawImage(
        scale=obs_cond.pixel_scale,
        nx=psf_stamp_size,
        ny=psf_stamp_size).array
    return psf_image, mean_sky_level


class Stack_params(Measurement_params):
    """Class with functions that describe how LSST science pipeline can
    perform measurements on the input data."""
    min_pix = 1  # Minimum size in pixels to be considered a source
    bkg_bin_size = 32  # Binning size of the local background
    thr_value = 5  # SNR threshold for the detection
    psf_stamp_size = 41  # size of psf stamp to draw PSF on

    def make_measurement(self, data, index):
        """Perform detection, deblending and measurement on the i band image of
        the blend for input index entry in the batch.

        Args:
            data: Dictionary with blend images, isolated object images, blend
                catalog, and observing conditions.
            index: Position of the blend to measure in the batch.

        Returns:
            astropy.Table of the measurement results.
         """
        image_array = data['blend_images'][index, :, :, 3].astype(np.float32)
        psf_image, mean_sky_level = get_psf_sky(
            data['obs_condition'][index][3], self.psf_stamp_size)
        variance_array = image_array + mean_sky_level
        psf_array = psf_image.astype(np.float64)
        cat = run_stack(image_array, variance_array, psf_array,
                        min_pix=self.min_pix, bkg_bin_size=self.bkg_bin_size,
                        thr_value=self.thr_value)
        cat_chldrn = cat[cat['deblend_nChild'] == 0]
        cat_chldrn = cat_chldrn.copy(deep=True)
        return cat_chldrn.asAstropy()

    def get_deblended_images(self, data, index):
        return None


def run_stack(image_array, variance_array, psf_array,
              min_pix=1, bkg_bin_size=32, thr_value=5):
    """
    Function to setup the DM stack and perform detection, deblending and
    measurement
    Args:
        image_array: Numpy array of image to run stack on
        variance_array: per pixel variance of the input image_array (must
            have same dimensions as image_array)
        psf_array: Image of the PSF for image_array.
        min_pix: Minimum size in pixels of a source to be considered by the
            stack (default=1).
        bkg_bin_size: Binning of the local background in pixels (default=32).
        thr_value: SNR threshold for the detected sources to be included in the
            final catalog(default=5).
    Returns:
        catalog: AstroPy table of detected sources
    """
    # Convert to stack Image object
    import lsst
    import lsst.afw.image
    import lsst.afw.math
    import lsst.meas.base
    import lsst.meas.algorithms
    import lsst.afw.table
    import lsst.meas.deblender
    import lsst.afw.table

    image = lsst.afw.image.ImageF(image_array)
    variance = lsst.afw.image.ImageF(variance_array)
    # Generate a masked image, i.e., an image+mask+variance image (mask=None)
    masked_image = lsst.afw.image.MaskedImageF(image, None, variance)
    # Create the kernel in the stack's format
    psf_im = lsst.afw.image.ImageD(psf_array)
    fkernel = lsst.afw.math.FixedKernel(psf_im)
    psf = lsst.meas.algorithms.KernelPsf(fkernel)
    # Passing the image to the stack
    exposure = lsst.afw.image.ExposureF(masked_image)
    # Assign the exposure the PSF that we created
    exposure.setPsf(psf)
    schema = lsst.afw.table.SourceTable.makeMinimalSchema()
    config1 = lsst.meas.algorithms.SourceDetectionConfig()
    # Tweaks in the configuration that can improve detection
    # Change carefully!
    #####
    config1.tempLocalBackground.binSize = bkg_bin_size
    config1.minPixels = min_pix
    config1.thresholdValue = thr_value
    #####
    detect = lsst.meas.algorithms.SourceDetectionTask(schema=schema,
                                                      config=config1)
    deblend = lsst.meas.deblender.SourceDeblendTask(schema=schema)
    config1 = lsst.meas.base.SingleFrameMeasurementConfig()
    # config1.plugins.names.add('ext_shapeHSM_HsmShapeRegauss')
    # config1.plugins.names.add('ext_shapeHSM_HsmSourceMoments')
    # config1.plugins.names.add('ext_shapeHSM_HsmPsfMoments')
    measure = lsst.meas.base.SingleFrameMeasurementTask(schema=schema,
                                                        config=config1)
    table = lsst.afw.table.SourceTable.make(schema)
    detect_result = detect.run(table, exposure)  # run detection task
    catalog = detect_result.sources
    deblend.run(exposure, catalog)  # run the deblending task
    measure.run(catalog, exposure)  # run the measuring task
    catalog = catalog.copy(deep=True)
    return catalog


class Scarlet_params(Measurement_params):
    """"""
    iters = 200  # Maximum number of iterations for scarlet to run
    e_rel = 1e-4  # Relative error for convergence
    detect_centers = True

    def __init__(self, show_scene=False):
        """Class with functions that describe how scarlet should deblend
        images in the input data

        Args:
            show_scene: If True plot the scarlet deblended model and residual
                image (default is False).
        """
        self.show_scene = show_scene

    def get_centers(self, image):
        """Returns centers from SEP detection on the band averaged mean of the
        input image.

        Args:
            image: Numpy array of multi-band image to run scarlet on
                [Number of bands, height, width].

        Returns:
            Array of x and y coordinate of centroids of objects in the image.
        """
        sep = __import__('sep')
        detect = image.mean(axis=0)  # simple average for detection
        bkg = sep.Background(detect)
        catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
        return np.stack((catalog['x'], catalog['y']), axis=1)

    def scarlet_initialize(self, images, peaks, psfs,
                           variances, bands):
        """ Initializes scarlet ExtendedSource at locations specified as
        peaks in the (multi-band) input images.
        Args:
            images: Numpy array of multi-band image to run scarlet on
                [Number of bands, height, width].
            peaks: Array of x and y coordinate of centroids of objects in
                the image [number of sources, 2].
            psfs: Numpy array of psf image in all bands [Number of bands,
                height, width].
            variances: Variance image of the blend scene[Number of bands,
                height, width].
            bands: List of filter names in which to simulate images.

        Returns:
            blend: scarlet.Blend object for the initialized sources
            observation: scarlet.Observation object with information to render
                the scarlet model.
        """
        scarlet = __import__("scarlet")
        model_psf = scarlet.PSF(
            partial(scarlet.psf.gaussian, sigma=0.8),
            shape=(None, 41, 41))
        model_frame = scarlet.Frame(
            images.shape,
            psfs=model_psf,
            channels=bands)
        observation = scarlet.Observation(
            images,
            psfs=scarlet.PSF(psfs),
            weights=1. / variances,
            channels=bands).match(model_frame)
        sources = []
        for n, peak in enumerate(peaks):
            result = scarlet.ExtendedSource(
                model_frame, (peak[1], peak[0]), observation,
                symmetric=True, monotonic=True,
                thresh=1, shifting=True)
            sources.append(result)
        blend = scarlet.Blend(sources, observation)
        blend.fit(self.iters, e_rel=self.e_rel)
        if self.show_scene:
            n_sources = len(list(sources))
            plot_utils.show_scarlet_residual(
                n_sources, blend, observation=observation, limits=(30, 90))
        return blend, observation

    def get_deblended_images(self, data, index):
        """Deblend input images with scarlet.

        Args:
            data (dict): Output generated by btk.draw_blends containing blended
                images, isolated images, observing conditions and blend
                catalog, for a given batch.
            index (int): Index number of blend scene in the batch to preform
                measurement on.

        Returns:
            a dict with the scarlet deblended images and peaks of the sources.
        """
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        bands = []
        psf_stamp_size = 41
        psfs = np.zeros((len(images), psf_stamp_size, psf_stamp_size),
                        dtype=np.float32)
        variances = np.zeros_like(images)
        n_bands = images.shape[0]
        for i in range(n_bands):
            bands.append(data['obs_condition'][index][i].filter_band)
            psf, mean_sky_level = get_psf_sky(
                data['obs_condition'][index][i], psf_stamp_size)
            psfs[i] = psf
            variances[i] = images[i] + mean_sky_level
        blend_cat = data['blend_list'][index]
        if self.detect_centers:
            peaks = self.get_centers(images)
        else:
            peaks = np.stack((blend_cat['dx'], blend_cat['dy']), axis=1)
        blend, observation = self.scarlet_initialize(
            images, peaks, psfs, variances, np.array(bands, dtype=str))
        im, selected_peaks = [], []
        for k, component in enumerate(blend):
            y, x = component.center
            selected_peaks.append([x, y])
            model = component.get_model()
            model_ = observation.render(model)
            im.append(np.transpose(model_, axes=(1, 2, 0)))
        return {'deblend_image': np.array(im), 'peaks': selected_peaks}


def make_true_seg_map(image, threshold):
    """Returns a boolean segmentation map corresponding to pixels in
    image above a certain threshold value.
    Args:
        image: Image to estimate segmentation map of
        threshold: Pixels above this threshold are marked as belonging to
            segmentation map

    Returns:
        Boolean segmentation map of the image
    """
    seg_map = np.zeros_like(image)
    seg_map[image < threshold] = 0
    seg_map[image >= threshold] = 1
    return seg_map.astype(np.bool)


def basic_selection_function(catalog):
    """Apply selection cuts to the input catalog.

    Only galaxies that satisfy the below criteria are returned:
    1) i band magnitude less than 27
    2) Second moment size is less than 4 arcsec.
    Second moments size (r_sec) computed as described in A1 of Chang et.al 2012

    Args:
        catalog: CatSim-like catalog from which to sample galaxies.

    Returns:
        CatSim-like catalog after applying selection cuts.
    """
    f = catalog['fluxnorm_bulge'] / (catalog['fluxnorm_disk'] +
                                     catalog['fluxnorm_bulge'])
    r_sec = np.hypot(catalog['a_d'] * (1 - f) ** 0.5 * 4.66,
                     catalog['a_b'] * f ** 0.5 * 1.46)
    q, = np.where((r_sec <= 4) & (catalog['i_ab'] <= 27))
    return catalog[q]


def basic_sampling_function(Args, catalog):
    """Samples galaxies from input catalog to make blend scene.

    Then number of galaxies in a blend are drawn from a uniform
    distribution of one up to Args.max_number. Function always selects one
    bright galaxy that is less than 24 imag. The other galaxies are selected
    from a sample with i<25.3 90% of the times and the remaining 10% with i<28.
    All galaxies must have semi-major axis is between 0.2 and 2 arcsec.
    The centers are randomly distributed within 1/30 *sqrt(N) of the postage
    stamp size, where N is the number of objects in the blend.

    Args:
        Args: Class containing input parameters.
        catalog: CatSim-like catalog from which to sample galaxies.

    Returns:
        Catalog with entries corresponding to one blend.
    """
    number_of_objects = np.random.randint(0, Args.max_number)
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    cond = (a <= 2) & (a > 0.2)
    q_bright, = np.where(cond & (catalog['i_ab'] <= 24))
    if np.random.random() >= 0.9:
        q, = np.where(cond & (catalog['i_ab'] < 28))
    else:
        q, = np.where(cond & (catalog['i_ab'] <= 25.3))
    blend_catalog = astropy.table.vstack(
        [catalog[np.random.choice(q_bright, size=1)],
         catalog[np.random.choice(q, size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    # keep number density of objects constant
    maxshift = Args.stamp_size / 30. * number_of_objects ** 0.5
    dx, dy = btk.create_blend_generator.get_random_center_shift(
        Args, number_of_objects + 1, maxshift=maxshift)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    return blend_catalog


def group_sampling_function(Args, catalog):
    """Blends are defined from *groups* of galaxies from the CatSim
    catalog previously analyzed with WLD.

    The group is centered on the middle of the postage stamp.
    Function only draws galaxies that lie within the postage stamp size
    determined in Args. The name of the pre-run wld catalog must be defined as
    Args.wld_catalog_name.

    Note: the pre-run WLD images are not used here. We only use the pre-run
    catalog (in i band) to identify galaxies that belong to a group.

    Args:
        Args: Class containing input parameters.
        catalog: CatSim-like catalog from which to sample galaxies.

    Returns:
        Catalog with entries corresponding to one blend.
    """
    if not hasattr(Args, 'wld_catalog_name'):
        raise Exception("A pre-run WLD catalog  name should be input as "
                        "Args.wld_catalog_name")
    else:
        wld_catalog = astropy.table.Table.read(Args.wld_catalog_name,
                                               format='fits')
    # randomly sample a group.
    group_ids = np.unique(wld_catalog['grp_id'][wld_catalog['grp_size'] >= 2])
    group_id = np.random.choice(group_ids, replace=False)
    # get all galaxies belonging to the group.
    ids = wld_catalog['db_id'][wld_catalog['grp_id'] == group_id]
    blend_catalog = astropy.table.vstack(
        [catalog[catalog['galtileid'] == i] for i in ids])
    # Set mean x and y coordinates of the group galaxies to the center of the
    # postage stamp.
    blend_catalog['ra'] -= np.mean(blend_catalog['ra'])
    blend_catalog['dec'] -= np.mean(blend_catalog['dec'])
    # convert ra dec from degrees to arcsec
    blend_catalog['ra'] *= 3600
    blend_catalog['dec'] *= 3600
    # Add small random shift so that center does not perfectly align with
    # the stamp center
    dx, dy = btk.create_blend_generator.get_random_center_shift(
        Args, 1, maxshift=3 * Args.pixel_scale)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    # make sure galaxy centers don't lie too close to edge
    cond1 = np.abs(blend_catalog['ra']) < Args.stamp_size / 2. - 3
    cond2 = np.abs(blend_catalog['dec']) < Args.stamp_size / 2. - 3
    no_boundary = blend_catalog[cond1 & cond2]
    if len(no_boundary) == 0:
        return no_boundary
    # make sure number of galaxies in blend is less than Args.max_number
    # randomly select max_number of objects if larger.
    num = min([len(no_boundary), Args.max_number])
    select = np.random.choice(range(len(no_boundary)), num, replace=False)
    return no_boundary[select]


def group_sampling_function_numbered(Args, catalog):
    """Blends are defined from *groups* of galaxies from a CatSim-like
    catalog previously analyzed with WLD.

    This function requires a parameter, group_id_count, to be input in Args
    along with the wld_catalog which tracks the group id returned. Each time
    the generator is called, 1 gets added to the count. If the count is
    larger than the number of groups input, the generator is forced to exit.

    The group is centered on the middle of the postage stamp.
    This function only draws galaxies whose centers lie within 1 arcsec the
    postage stamp edge, which may cause the number of galaxies in the blend to
    be smaller than the group size. The pre-run wld catalog must be defined as
    Args.wld_catalog.

    Note: the pre-run WLD images are not used here. We only use the pre-run
    catalog (in i band) to identify galaxies that belong to a group.

    Args:
        Args: Class containing input parameters.
        catalog: CatSim-like catalog from which to sample galaxies.

    Returns:
        Catalog with entries corresponding to one blend.
    """
    if not hasattr(Args, 'wld_catalog'):
        raise Exception(
            "A pre-run WLD catalog should be input as Args.wld_catalog")
    if not hasattr(Args, 'group_id_count'):
        raise NameError("An integer specifying index of group_id to draw must"
                        "be input as Args.group_id_count")
    elif not isinstance(Args.group_id_count, int):
        raise ValueError("group_id_count must be an integer")
    # randomly sample a group.
    group_ids = np.unique(
        Args.wld_catalog['grp_id'][
            (Args.wld_catalog['grp_size'] >= 2) &
            (Args.wld_catalog['grp_size'] <= Args.max_number)])
    if Args.group_id_count >= len(group_ids):
        message = "group_id_count is larger than number of groups input"
        raise GeneratorExit(message)
    else:
        group_id = group_ids[Args.group_id_count]
        Args.group_id_count += 1
    # get all galaxies belonging to the group.
    # make sure some group or galaxy was not repeated in wld_catalog
    ids = np.unique(
        Args.wld_catalog['db_id'][Args.wld_catalog['grp_id'] == group_id])
    blend_catalog = astropy.table.vstack(
        [catalog[catalog['galtileid'] == i] for i in ids])
    # Set mean x and y coordinates of the group galaxies to the center of the
    # postage stamp.
    blend_catalog['ra'] -= np.mean(blend_catalog['ra'])
    blend_catalog['dec'] -= np.mean(blend_catalog['dec'])
    # convert ra dec from degrees to arcsec
    blend_catalog['ra'] *= 3600
    blend_catalog['dec'] *= 3600
    # Add small random shift so that center does not perfectly align with stamp
    # center
    dx, dy = btk.create_blend_generator.get_random_center_shift(
        Args, 1, maxshift=5 * Args.pixel_scale)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    # make sure galaxy centers don't lie too close to edge
    cond1 = np.abs(blend_catalog['ra']) < Args.stamp_size / 2. - 1
    cond2 = np.abs(blend_catalog['dec']) < Args.stamp_size / 2. - 1
    no_boundary = blend_catalog[cond1 & cond2]
    message = ("Number of galaxies greater than max number of objects per"
               f"blend. Found {len(no_boundary)}, expected <= {Args.max_number}")
    assert len(no_boundary) <= Args.max_number, message
    return no_boundary


class Basic_measure_params(Measurement_params):
    """Class to perform detection by identifying peaks with skimage"""

    def get_centers(self, image):
        """Return centers detected when object detection is performed on the
        input image with skimage.feature.peak_local_max.

        Args:
            image (float): Image (single band) of galaxy to perform measurement

        Returns:
                centers: x and y coordinates of detected  centroids
        """
        # set detection threshold to 5 times std of image
        threshold = 5 * np.std(image)
        coordinates = skimage.feature.peak_local_max(image, min_distance=2,
                                                     threshold_abs=threshold)
        return np.stack((coordinates[:, 1], coordinates[:, 0]), axis=1)

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend and centers for the given blend"""
        image = np.mean(data['blend_images'][index], axis=2)
        peaks = self.get_centers(image)
        return {'deblend_image': None, 'peaks': peaks}


class Basic_metric_params(Metrics_params):
    """Class describing functions to return results of
    detection/deblending/measurement algorithm in meas_generator. Each
    time the algorithm is called, it is run on a batch of blends yielded
    by the meas_generator.
    """

    def get_detections(self):
        """Returns input blend catalog and detection catalog for
        the detection performed.

        Returns:
            Results of the detection algorithm are returned as:
                true_tables: List of astropy Table of the blend catalogs of the
                    batch. Length of tables must be the batch size. x and y
                    coordinate values must be under columns named 'dx' and 'dy'
                    respectively, in pixels from bottom left corner as (0, 0).
                detected_tables: List of astropy Table of output from detection
                    algorithm. Length of tables must be the batch size. x and y
                    coordinate values must be under columns named 'dx' and 'dy'
                    respectively, in pixels from bottom left corner as (0, 0).
        """
        blend_op, deblend_op, _ = next(self.meas_generator)
        true_tables = blend_op['blend_list']
        detected_tables = []
        for i in range(len(true_tables)):
            detected_centers = deblend_op[i]['peaks']
            detected_table = astropy.table.Table(detected_centers,
                                                 names=['dx', 'dy'])
            detected_tables.append(detected_table)
        return true_tables, detected_tables


class Stack_metric_params(Metrics_params):
    """Class describing functions to return results of
    detection/deblending/measurement algorithm in meas_generator.  Each
    time the algorithm is called, it is run on a batch of blends yielded
    by the meas_generator.
    """

    def get_detections(self):
        """Returns blend catalog and detection catalog for detection performed.

        Returns:
            Results of the detection algorithm are returned as:
                true_tables: List of astropy Table of the blend catalogs of the
                    batch. Length of tables must be the batch size. x and y
                    coordinate values must be under columns named 'dx' and 'dy'
                    respectively, in pixels from bottom left corner as (0, 0).
                detected_tables: List of astropy Table of output from detection
                    algorithm. Length of tables must be the batch size. x and y
                    coordinate values must be under columns named 'dx' and 'dy'
                    respectively, in pixels from bottom left corner as (0, 0).
        """
        blend_op, _, cat = next(self.meas_generator)
        # Get astropy table with entries corresponding to true sources
        true_tables = blend_op['blend_list']
        detected_tables = []
        for i in range(len(true_tables)):
            detected_centers = np.stack(
                [cat[i]['base_NaiveCentroid_x'],
                 cat[i]['base_NaiveCentroid_y']],
                axis=1)
            detected_table = astropy.table.Table(detected_centers,
                                                 names=['dx', 'dy'])
            detected_tables.append(detected_table)
        return true_tables, detected_tables


def get_detection_eff_matrix(summary_table, num):
    """Computes the detection efficiency matrix for the input detection summary
    table.

    Input argument num sets the maximum number of true objects per blend in the
    test set for which the
    detection efficiency matrix is to be created for. Detection efficiency is
    computed for a number of true objects in the range (0-num) as columns and
    the detection percentage as rows. The percentage values in a column sum to
    100.

    The input summary table must be a numpy array of shape [N, 5], where N is
    the test set size. The 5 columns in the summary_table are number of true
    objects, detected sources, undetected objects, spurious detections and
    shredded objects for each of the N blend scenes in the test set.

    Args:
        summary_table (`numpy.array`): Detection summary as a table [N, 5].
        num (int): Maximum number of true objects to create matrix for. Number
            of columns in efficiency matrix will be num+1. The first column
            will correspond to no true objects.

    Returns:
        numpy.ndarray of size[num+2, num+1] that shows detection efficiency.
    """
    eff_matrix = np.zeros((num + 2, num + 1))
    for i in range(0, num + 1):
        q_true, = np.where(summary_table[:, 0] == i)
        for j in range(0, num + 2):
            if len(q_true) > 0:
                q_det, = np.where(summary_table[q_true, 1] == j)
                eff_matrix[j, i] = len(q_det)
    norm = np.sum(eff_matrix, axis=0)
    # If no detections along a column, set sum to 1 to avoid dividing by zero.
    norm[norm == 0.] = 1
    # normalize over columns.
    eff_matrix = eff_matrix / norm[np.newaxis, :] * 100.
    return eff_matrix
