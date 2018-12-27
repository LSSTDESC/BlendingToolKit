"""Contains functions to perform detection, deblending and measurement
    on images.
"""
from btk import measure
import numpy as np


class SEP_params(measure.Measurement_params):
    """Class to perform detection and deblending with SEP"""
    def get_centers(self, image):
        """Return centers detected when object detection and photometry
        is done on input image with SEP.
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
        """Returns scarlet modeled blend  and centers for the given blend"""
        image = np.mean(data['blend_images'][index], axis=2)
        peaks = self.get_centers(image)
        return [None, peaks]


class Stack_params(measure.Measurement_params):
    min_pix = 1
    bkg_bin_size = 32
    thr_value = 5
    psf_stamp_size = 41

    def get_psf_sky(self, obs_cond):
        mean_sky_level = obs_cond.mean_sky_level
        psf = obs_cond.psf_model
        psf_image = psf.drawImage(
           nx=self.psf_stamp_size,
           ny=self.psf_stamp_size).array
        return psf_image, mean_sky_level

    def make_measurement(self, data, index):
        """Perform detection, deblending and measurement on the i band image of
        the blend image for input index in the batch.
         """
        image_array = data['blend_images'][index, :, :, 3].astype(np.float32)
        psf_image, mean_sky_level = self.get_psf_sky(data['obs_condition'][3])
        variance_array = image_array + mean_sky_level
        psf_array = psf_image.astype(np.float64)
        cat = run_stack(image_array, variance_array, psf_array,
                        min_pix=self.min_pix, bkg_bin_size=self.bkg_bin_size,
                        thr_value=self.thr_value)
        cat_chldrn = cat[cat['deblend_nChild'] == 0]
        cat_chldrn = cat_chldrn.copy(deep=True)
        return cat_chldrn.asAstropy()

    def get_deblended_images(self, data=None, index=None):
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
    import lsst.afw.table
    import lsst.afw.image
    import lsst.afw.math
    import lsst.meas.algorithms
    import lsst.meas.base
    import lsst.meas.deblender
    import lsst.meas.extensions.shapeHSM
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
    config1.plugins.names.add('ext_shapeHSM_HsmShapeRegauss')
    config1.plugins.names.add('ext_shapeHSM_HsmSourceMoments')
    config1.plugins.names.add('ext_shapeHSM_HsmPsfMoments')
    measure = lsst.meas.base.SingleFrameMeasurementTask(schema=schema,
                                                        config=config1)
    table = lsst.afw.table.SourceTable.make(schema)
    detect_result = detect.run(table, exposure)  # run detection task
    catalog = detect_result.sources
    deblend.run(exposure, catalog)  # run the deblending task
    measure.run(catalog, exposure)  # run the measuring task
    catalog = catalog.copy(deep=True)
    return catalog


class Scarlet_params(measure.Measurement_params):
    iters = 200
    e_rel = .015
    detect_centers = True

    def make_measurement(self, data=None, index=None):
        return None

    def get_centers(self, image):
        sep = __import__('sep')
        detect = image.mean(axis=0)  # simple average for detection
        bkg = sep.Background(detect)
        catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
        return np.stack((catalog['x'], catalog['y']), axis=1)

    def scarlet_initialize(self, images, peaks,
                           bg_rms, iters, e_rel):
        """ Initializes scarlet ExtendedSource at locations specified as
        peaks in the (multi-band) input images.
        Args:
            images: Numpy array of multi-band image to run scarlet on
                    [Number of bands, height, width].
            peaks: Array of x and y coordinate of centroids of objects in
                   the image [number of sources, 2].
            bg_rms: Background RMS value of the images [Number of bands]
        Returns
            blend: scarlet.Blend object for the initialized sources
            rejected_sources: list of sources (if any) that scarlet was
                              unable to initialize the image with.
        """
        scarlet = __import__("scarlet")
        sources, rejected_sources = [], []
        for n, peak in enumerate(peaks):
            try:
                result = scarlet.ExtendedSource(
                    (peak[1], peak[0]),
                    images,
                    bg_rms)
                sources.append(result)
            except scarlet.source.SourceInitError:
                rejected_sources.append(n)
                print("No flux in peak {0} at {1}".format(n, peak))
        blend = scarlet.Blend(sources).set_data(images, bg_rms=bg_rms)
        blend.fit(iters, e_rel=e_rel)
        return blend, rejected_sources

    def get_deblended_images(self, data, index):
        """
        Deblend input images with scarlet
        Args:
        images: Numpy array of multi-band image to run scarlet on
               [Number of bands, height, width].
        peaks: Array of x and y coordinate of centroids of objects in the image.
               [number of sources, 2]
        bg_rms: Background RMS value of the images [Number of bands]
        iters: Maximum number of iterations if scarlet doesn't converge
               (Default: 200).
        e_rel: Relative error for convergence (Default: 0.015)
        Returns
        blend: scarlet.Blend object for the initialized sources
        rejected_sources: list of sources (if any) that scarlet was
        unable to initialize the image with.
        """
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        blend_cat = data['blend_list'][index]
        if self.detect_centers:
            peaks = self.get_centers(images)
        else:
            peaks = np.stack((blend_cat['dx'], blend_cat['dy']), axis=1)
        bg_rms = np.array(
            [data['obs_condition'][i].mean_sky_level**0.5 for i in range(len(images))])
        blend, rejected_sources = self.scarlet_initialize(images, peaks,
                                                          bg_rms, self.iters,
                                                          self.e_rel)
        im, selected_peaks = [], []
        for m in range(len(blend.sources)):
            im .append(np.transpose(blend.get_model(k=m), axes=(1, 2, 0)))
            selected_peaks.append(
                [blend.components[m].center[1], blend.components[m].center[0]])
        return [np.array(im), selected_peaks]


def make_true_seg_map(image, threshold):
    """Returns a boolean segmentation map corresponding to pixels in
    image above a certain threshold value.threshold
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
