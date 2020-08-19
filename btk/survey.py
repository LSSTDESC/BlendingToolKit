import descwl


class Survey(descwl.survey.Survey):
    """ Extension of the descwl survey class including information for the WCS
        
        Args:
            center_pix : tuple representing the center of the image in pixels 
            center_sky : tuple representing the center of the image in sky coordinates
                         (RA,DEC) in arcseconds.
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
