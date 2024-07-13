btk.survey module
=========================
The survey module interfaces with the `surveycodex <https://github.com/LSSTDESC/surveycodex>`_ package.

This package contains various informations about different surveys, including `LSST`, `HSC` (wide-field survey), HST COSMOS, etc. These informations are stored in a surveycodex :class:`~surveycodex.survey.Survey` object, containing several surveycodex :class:`~surveycodex.filter.Filter` objects, which is retrieved in BTK to compute fluxes and noise levels (using functions which are available in surveycodex).

The main modifications made by BTK to the surveycodex objects are adding the PSF to the Filter object, and making them editable. This is achieved by defining a BTK :class:`~bkt.survey.Survey` and a BTK :class:`~bkt.survey.Filter` objects, which directly inherit from the corresponding surveycodex objects.

.. automodule:: btk.survey
