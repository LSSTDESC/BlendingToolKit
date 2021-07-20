CLI
=============


Introduction
----------------------

The CLI is designed to run BTK from start to end, starting by creating a ``DrawBlendGenerator``, followed by a ``MeasureGenerator``, and finally calculating and saving the corresponding metrics via a ``MetricsGenerator``.

Assuming that BTK has been pip installed, you can run btk from the command line like:

.. code-block::
  btk sampling=default draw_blends=catsim max_number=3 save_path=/directory/to/save/results cpus=1
    verbose=False surveys=[Rubin, HST] surveys.Rubin.airmass=1.1
    sampling=default catalog.name=catsim use_metrics=['detection', 'segmentation'] (...)

You need to create the directory to save results yourself (preferably an empty directory) and specify its absolute path when you run the CLI via the ``save_path`` argument.

*Important:* To use the CLI you first need to setup an environment variable that contains the
absolute path to your BTK local repo:

.. code-block::
    export BTK_HOME=/path/to/local/btk/repo

Available options
----------------------

Here is a list of all the options of BTK you can customize directly from the CLI.

* sampling: Specify the sampling function to be used, options:

    - default

    - galsim_hub

    - group_sampling

    - group_sampling_numbered

* catalog: Attribute group consisting of two sub-attributes.

    - catalog.name: Name of the BTK catalog class, options: {catsim, cosmos}

    - catalog.catalog_files: Path to files containing catalog information. The 'catsim'
          catalog requires one path, while the 'cosmos' type requires two paths specified as
          a list. See the 'Using COSMOS galaxies' section in the tutorials page for more details.

* surveys: Name of the survey(s) you want to use, options are:

      - Rubin

      - HST

      - HSC

      - DES

      - CFHT

      - Euclid

      and correspond to each of the config files available in ``conf/surveys``. You can pass in a list of surveys for multi-resolution
      studies too. For example:

      .. code-block::
          btk surveys=[Rubin, HST] (...)

      Assuming that you want to use e.g. the Rubin survey default parameters but some changes you can modify individual parameters of this survey directly from the
      command line:

      .. code-block::
          btk surveys=Rubin surveys.Rubin.airmass=1.1 (...)

      If you want to modify a large number of parameters of a given survey, it might be easier to
      add your own config file to ``conf/surveys``. See below for instructions on how to do this.

* draw_blends: Which draw_blend_generator to use, options are

    - catsim

    - cosmos

    - galsim_hub


* save_path: Absolute path to a (preferably) empty directory where you would like to save
    results of running BTK.

* cpus: Number of cpus you would like to use for multiprocessing.

* verbose: Whether you would like BTK to print

* batch_size: Size of the batches produced by the various BTK generators.

* stamp_size: Stamp size of images in arcseconds.

* max_shift: Maximum shift of galaxies from the center of each postage stamp in arcseconds.

* add_noise: Whether to add (Poisson) noise to the images.

* channels_last: Whether to use have images in channel last format (True) or not (False).

* galsim_hub: Attribute group consisting of options:

    - model: Which galsim_hub model to use (default: 'hub:Lanusse2020')

    - param_names: list of the parameters with which the generation is parametrized; this is
        unique to each model (default: ``['flux_radious, 'mag_auto', 'zphot']``).
        NOTE: This flag is ignored unless ``draw_blends=galsim_hub``

* measure_kwargs: Dictionary or list of dictionaries containing the keyword arguments to be
    passed in to each measure_function.

* measure_functions: List of measure_functions to be ran, options:

    - basic

    - sep

* meas_band_num: Band index to perform measurements in.

* noise_threshold_factor: Factor for determining the threshold which is applied when
    getting segmentations from true images. A value of ``3`` would correspond to a threshold of
    three sigmas (with sigma the standard deviation of the noise)

* distance_threshold_match: Maximum distance for matching a detected and a true galaxy
    in pixels (default: ``5.0``).

Creating your own Survey
---------------------------

To create your own survey in BTK, the easiest way is to write your own yaml file that follows the
same structure as the other yaml file in `conf/surveys`. Note that the top-level dictionary key
needs to be the *unique* name of your survey and the corresponding fields

    - name

    - pixel_scale

    - effective_area

    - mirror_diameter

    - airmass

    - zeropoint_airmass

    - filters

are required. You should have at least one filter with the fields:

    - name

    - sky_brightness

    - exp_time

    - zeropoint

    - extinction

    - psf

The ``psf`` field can be specified with ``type: default`` in which case you need to specify the parameters:

    - fwhm

    - mirror_diameter

    - effective_area

    - filt_wavelength

in the ``params`` dictionary (see examples for how to reference already existing values in the
config file). The ``psf`` can  also be specified as ``type: galsim`` and you can provide the same format of a PSF as you would in a galsim config file (with no reference to external data sources).

CLI help
---------------------------
You can always access the help menu of the CLI if you forget any of the options like:

.. code-block::
    btk --help
