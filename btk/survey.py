import descwl

class Survey(descwl.survey.Survey):
	""" Extension of the descwl survey class including information for the WCS
		
	    Args:
			center_pix : tuple representing the center of the image in pixels 
			center_sky : tuple representing the center of the image in sky coordinates ( (RA,DEC) in arcseconds) 
			**args : any arguments given to a descwl survey
	"""
	def __init__(self, center_pix=None, center_sky=None, projection=None, **args):
		super(Survey, self).__init__(**args)
		self.center_pix = center_pix
		self.center_sky = center_sky
		self.projection = projection
