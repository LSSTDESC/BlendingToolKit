"""Contains functions to perform detection, deblending and measurement
    on images.
"""
import lsst.afw.table
import lsst.afw.image
import lsst.afw.math
import lsst.meas.algorithms
import lsst.meas.base
import lsst.meas.deblender
import lsst.meas.extensions.shapeHSM
import scarlet
from measure import Measurement_params


class Stack_params(Measurement_params):
    min_pix = 1
    bkg_bin_size = 32
    thr_value = 5

    def make_measurement(self, data):
        image_array, variance_array, psf_array = data
        cat = run_stack(image_array, variance_array, psf_array,
                        min_pix=self.min_pix, bkg_bin_size=self.bkg_bin_size,
                        thr_value=self.thr_value)
        mask = cat['deblend_nChild'] == 0
        mask &= cat['base_SdssCentroid_flag'] is False
        cat_chldrn = cat[mask]
        cat_chldrn = cat_chldrn.copy(deep=True)
        return cat_chldrn.asAstropy()

    def get_deblended_images(self, data):
        return None


def run_stack(image_array, variance_array, psf_array,
              min_pix=1, bkg_bin_size=32, thr_value=5):
    """
    Function to setup the DM stack and perform detection, deblending and
    measuremnt
    Args:
        image_array: Numpy array of image to run stack on
        variance_array: per pixel variance of the input image_array (must
                        have same dimensions as image_array)
        psf_array: Image of the psf for image_array.
        min_pix: Minimum size in pixels of a source to be considered by the
                 stack (default=1).
        bkg_bin_size: Binning of the local background in pixels (default=32).
        thr_value: SNR threshold for the detected sources to be included in the
                   final catalog(default=5).
    Returns:
        catalog: Astropy table of detected sources
    """
    # Convet to stack Image object
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


class Scarlet_params(Measurement_params):
    iters = 200
    e_rel = .015

    def make_measurement(self, data=None):
        return None

    def get_deblended_images(self, data):
        """
        Deblend input images with scarlet
        Args:
        images: Numpy array of multi-band image to run scarlet on
               [Number of bands, height, width].
        peaks: Array of x and y cordinate of cntroids of objects in the image.
               [number of sources, 2]
        bg_rms: Background RMS value of the images [Number of bands]
        iters: Maximum number of iterations if scarlet doesn't converge
               (Default: 200).
        e_rel: Relative error for convergence (Deafult: 0.015)
        Returns
        blend: scarlet.Blend object for the initialized sources
        rejected_sources: list of sources (if any) that scarlet was
        unable to initlaize the image with.
        """
        images, peaks, bg_rms = data
        blend, rejected_sources = scarlet_initialize(images, peaks,
                                                     bg_rms)
        sources, rejected_sources = [], []
        for n, peak in enumerate(peaks):
            try:
                result = scarlet.ExtendedSource((peak[1], peak[0]), images,
                                                bg_rms)
                sources.append(result)
            except scarlet.source.SourceInitError:
                rejected_sources.append(n)
                print("No flux in peak {0} at {1}".format(n, peak))
        blend = scarlet.Blend(sources).set_data(images, bg_rms=bg_rms)
        return blend, rejected_sources


def scarlet_initialize(images, peaks,
                       bg_rms):
    """ Intializes scarlet ExtendedSource at locations specified as peaks
    in the (multi-band) input images.
    Args:
        images: Numpy array of multi-band image to run scarlet on
               [Number of bands, height, width].
        peaks: Array of x and y cordinate of cntroids of objects in the image.
               [number of sources, 2]
        bg_rms: Background RMS value of the images [Number of bands]
    Returns
        blend: scarlet.Blend object for the initialized sources
        rejected_sources: list of sources (if any) that scarlet was
                          unable to initlaize the image with.
    """
    sources, rejected_sources = [], []
    for n, peak in enumerate(peaks):
        try:
            result = scarlet.ExtendedSource((peak[1], peak[0]), images,
                                            bg_rms)
            sources.append(result)
        except scarlet.source.SourceInitError:
            rejected_sources.append(n)
            print("No flux in peak {0} at {1}".format(n, peak))
    blend = scarlet.Blend(sources).set_data(images, bg_rms=bg_rms)
    return blend, rejected_sources
