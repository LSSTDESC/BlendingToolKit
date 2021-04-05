btk.survey module
=========================
The survey module defines the :class:`~btk.survey.Survey` class as a namedtuple, which contains the information relevant to observing conditions for a given survey. A Survey contains several instances of :class:`~btk.survey.Filter`, which are also namedtuples containing information specific to each filter.

.. automodule:: btk.survey
	:exclude-members: Survey, Filter

	.. autoclass:: Survey
		:exclude-members: airmass, filters, effective_area, mirror_diameter, zeropoint_airmass, pixel_scale, name, count, index

	.. autoclass:: Filter
		:exclude-members: exp_time, extinction, name, psf, sky_brightness, zeropoint, name, count, index
