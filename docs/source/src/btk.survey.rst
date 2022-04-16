btk.survey module
=========================
The survey module interfaces with the `galcheat <https://github.com/aboucaud/galcheat>`_ package ; this package contains various informations about different surveys, including LSST, HSC (wide-field survey), HST COSMOS... Those informations are stored in a :class:`~galcheat.survey.Survey` object, containing several `~galcheat.filter.Filter` objects, which is retrieved in BTK to compute fluxes and noise levels (using functions which are available in galcheat). The main modifications made by BTK to the galcheat objects are adding the PSF to the Filter object, and making them editable. This is achieved by defining a :class:`~bkt.survey.Survey` and a :class:`~bkt.survey.Filter` objects, which directly inherit from the galcheat objects.

.. automodule:: btk.survey
