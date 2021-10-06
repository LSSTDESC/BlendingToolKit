# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
"""Functions defining a light profile produced by a deep generative model.

Written by François Lanusse (@EiffL) as part of the galsim_hub package
https://github.com/McWilliamsCenter/galsim_hub

Modified by Alexandre Boucaud (@aboucaud) to adapt the code
to be used with TensorFlow2 and remove the tensorflow-hub dependency
for being used in BTK

"""
import galsim
import numpy as np
import tensorflow as tf

DEFAULT_PIXEL_SIZE = 0.1
DEFAULT_LATENT_SPACE_SIZE = 32


class GenerativeGalaxyModel:
    """Model for generating random galaxies."""

    def __init__(self, model_name, input_list, pixel_size=None, latent_size=None):
        """Initialisation of the galaxy generator, by loading a tensorflow model.

        Parameters
        ----------
        model_name : string
            Path to the tensorflow model (directory) to load,
            or a module name on galsim-hub using the format `hub:model_name`
        input_list : list of string
            Names of the physical quantities given as input of the generative model.
            The list can include "random_normal" to include a noise image
            in the processing.
        pixel_size : float, optional
            Size of the pixel on the sky in arcsec/pixel.
            This parameter should match the value for which the model has been
            trained for.
        latent_size : int, optional
            Size of the latent space of the generative model. This is only
            used for constructing the noise image when "random_normal" is
            among the inputs.

        """
        if model_name.startswith("hub:"):
            model_name = model_name.split(":")[1]
            model_name = f"https://github.com/McWilliamsCenter/galsim_hub/blob/master/hub/{model_name}/model.tar.gz?raw=true"  # noqa

        self.model_name = model_name
        self.pixel_size = pixel_size if pixel_size is not None else DEFAULT_PIXEL_SIZE
        self.latent_size = latent_size if latent_size is not None else DEFAULT_LATENT_SPACE_SIZE

        self.model = None

        self.quantities = [param for param in input_list if param != "random_normal"]
        self.random_variables = "random_normal" in input_list

    def prepare_inputs(self, catalogue, rng=None):
        """Cast the input catalog into a dictionary.

        Parameters
        ----------
        catalogue : table-like
            Table of input data
        rng : galsim.BaseDeviate, optional
            Galsim random generator

        Returns
        -------
        Dictionary with the inputs of the generator model

        """
        inputs = {key: tf.constant(catalogue[key], dtype=tf.float32) for key in self.quantities}

        if self.random_variables:
            batch_size = len(catalogue[self.quantities[0]])
            noise_shape = (batch_size, self.latent_size)
            noise_array = np.empty(noise_shape, dtype=np.float)
            # Draw a random normal from the galsim RNG
            # If not provided, create a RNG
            if rng is None:
                rng = galsim.BaseDeviate()
            galsim.random.GaussianDeviate(rng, sigma=1).generate(noise_array)

            inputs["random_normal"] = tf.constant(noise_array.astype("float32"), dtype=tf.float32)

        return inputs

    def sample(
        self,
        cat,
        noise=None,
        rng=None,
        x_interpolant=None,
        k_interpolant=None,
        pad_factor=4,
        noise_pad_size=0,
        gsparams=None,
    ):
        """Samples galaxy images from the model."""
        # If we are sampling for the first time
        if self.model is None:
            # Equivalent of tensorflow_hub.load()
            model = tf.saved_model.load(self.model_name)
            self.model = model.signatures["default"]

        # Populate feed dictionary with input data
        inputs = self.prepare_inputs(cat, rng)

        outputs = self.model(**inputs)["default"].numpy()

        # Now, we build an InterpolatedImage for each of these
        images = []
        for stamp in outputs:
            image = galsim.Image(
                np.ascontiguousarray(stamp.squeeze().astype(np.float64)), scale=self.pixel_size
            )

            images.append(
                galsim.InterpolatedImage(
                    image,
                    x_interpolant=x_interpolant,
                    k_interpolant=k_interpolant,
                    pad_factor=pad_factor,
                    noise_pad_size=noise_pad_size,
                    noise_pad=noise,
                    rng=rng,
                    gsparams=gsparams,
                )
            )

        if len(images) == 1:
            images = images[0]

        return images


if __name__ == "__main__":
    input_list = ["mag_auto", "zphot", "flux_radius"]
    pixel_size = 0.03
    cat = {"mag_auto": np.ones(4) * 20, "zphot": np.ones(4), "flux_radius": 10 * np.ones(4)}
    generator = GenerativeGalaxyModel("hub:Lanusse2020", input_list, pixel_size)
    generator.sample(cat)
