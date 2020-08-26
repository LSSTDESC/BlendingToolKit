import descwl

surveys = {
    "LSST": {"bands": ("y", "z", "i", "r", "g", "u"), "pixel_scale": 0.2},
    "DES": {"bands": ("i", "r", "g", "z"), "pixel_scale": 0.263},
    "CFHT": {"bands": ("i", "r"), "pixel_scale": 0.185},
    "HSC": {"bands": ("y", "z", "i", "r", "g",), "pixel_scale": 0.17},
}


class Survey(descwl.survey.Survey):
    """ Extension of the descwl survey class including information for the WCS
        
        Args:
            center_pix : tuple representing the center of the image in pixels 
            center_sky : tuple representing the center of the image in sky coordinates
                         (RA,DEC) in arcseconds.
            projection : string representing the type of projection for the WCS. If None,
                         it will default to "TAN". A list of available projections can
                         be found in the documentation of `astropy.wcs`
            wcs : an `astropy.wcs.wcs` object corresponding to the parameters center_pix,
                  center_sky, projection, pixel_scale and stamp_size.
            **kwargs : any arguments given to a descwl survey
    """

    def __init__(
        self, center_pix=None, center_sky=None, projection=None, wcs=None, **kwargs
    ):
        super(Survey, self).__init__(**kwargs)
        self.center_pix = center_pix
        self.center_sky = center_sky
        self.projection = projection
        self.wcs = wcs
