"""Running btk with input config file

This script contains functions that can parse the input config file containing
parameters to generate blend scene images with btk. The images are simulated
and analyzed with the user specified detection/deblending/measurement algorithm
specified in the config file.

At present the script can be run for three kinds of simulations:
    'two_gal': Two-galaxy blends sampled randomly from CatSim galaxies
    'multi_gal': Up to 10 galaxy blends sampled randomly from CatSim galaxies
    'group': Blends defined as galaxy “groups” from a pre-processed wld output

An example config file (input/example-config.yaml) shows the parameters for
these simulations.

"""

import importlib.util
import multiprocessing
import os
import subprocess
import sys
import types

import dill
import numpy as np
import yaml

import btk


def parse_config(config_gen, simulation, verbose):
    """Parses the config generator with the config yaml file information.

    Args:
        config_gen: Generator with values i the yaml config file.
        simulation: Name of simulation to run btk test on.
        verbose: If True prints description at multiple steps.

    Returns:
        dictionary with basic user input and parameter values to generate
        simulations set by simulation
    """
    config_dict = {'user_input': None, 'simulation': {}}
    for doc in config_gen:
        if 'user_input' in doc.keys():
            config_dict['user_input'] = doc['user_input']
            # if no user input utils file, then use default in btk
            if config_dict['user_input']['utils_filename'] == 'None':
                config_dict['user_input']['utils_filename'] = os.path.join(
                    os.path.dirname(btk.__file__), 'utils.py')
            continue
        if 'simulation' in doc.keys():
            if simulation == doc['simulation'] or simulation == 'all':
                config_dict['simulation'][doc['simulation']] = doc['config']
                if verbose:
                    print(f"{doc['simulation']} parameter values loaded to "
                          "config dict")
    if not config_dict['simulation']:
        raise ValueError('simulation name does not exist in config yaml file')
    return config_dict


def read_configfile(filename, simulation, verbose):
    """Reads config YAML file.

    Args:
        filename: YAML config file with configuration values to run btk.
        simulation: Name of simulation to run btk test on.
        verbose: If True prints description at multiple steps.

    Returns:
        dictionary with btk configuration.

    """
    with open(filename, 'r') as stream:
        config_gen = yaml.safe_load_all(stream)  # load all yaml files here
        if verbose:
            print(f"config file {filename} loaded")
        if not isinstance(config_gen, types.GeneratorType):
            raise ValueError("input config yaml file not should contain at  "
                             " dictionary with btk configuration.two documents"
                             ", one for user input, other denoting"
                             "simulation parameters")
        config_dict = parse_config(config_gen, simulation, verbose)
    return config_dict


def get_config_class(config_dict, catalog_name, verbose):
    """Simulation parameter values in the config dictionary are used to return a
    config.Simulation_param class.

    Args:
        config_dict (dict): dictionary with btk configuration.
        catalog_name (str): Name of the Catsim-like catalog to simulate objects
            from.
        verbose (bool): If True prints description at multiple steps.

    Returns:
        config.Simulation_param class object with values specified in
            the input dict.
    """
    config = btk.config.Simulation_params(catalog_name)
    for key, value in config_dict.items():
        if key in config.__dict__.keys():
            setattr(config, key, value)
    setattr(config, 'verbose', verbose)
    if isinstance(config_dict['add'], dict):
        for key, value in config_dict['add'].items():
            setattr(config, key, value)
    if verbose:
        config.display()
    return config


def get_catalog(param, user_config_dict,
                selection_function_name, verbose):
    """Returns catalog from which objects are simulated by btk

    Args:
        param: Class with btk simulation parameters.
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        selection_function_name (str): Name of the selection function in
            btk/utils.py.
        verbose (bool): If True prints description at multiple steps.

    Returns:
        `astropy.table.Table` with parameters corresponding to objects being
        simulated.
    """
    if selection_function_name != "None":
        try:
            utils_filename = user_config_dict['utils_filename']
            spec = importlib.util.spec_from_file_location(
                'select_utils', utils_filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules['select_utils'] = module
            spec.loader.exec_module(module)
            selection_function = getattr(module, selection_function_name)
        except AttributeError as e:
            print(e)
            utils_filename = os.path.join(
                os.path.dirname(btk.__file__), 'utils.py')
            spec = importlib.util.spec_from_file_location(
                'select_utils', utils_filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules['select_utils'] = module
            spec.loader.exec_module(module)
            selection_function = getattr(module, selection_function_name)
    else:
        utils_filename = os.path.join(
            os.path.dirname(btk.__file__), 'utils.py')
        selection_function = None
    catalog = btk.get_input_catalog.load_catalog(
        param, selection_function=selection_function)
    if verbose:
        print(f"Loaded {param.catalog_name} catalog with "
              f"{selection_function_name} selection "
              f"function defined in {utils_filename}")
    return catalog


def get_blend_generator(param, user_config_dict, catalog,
                        sampling_function_name, verbose):
    """Returns generator object that generates catalog describing blended
    objects.

    Args:
        param (class): Parameter values for btk simulations.
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        catalog: `astropy.table.Table` with parameters corresponding to objects
                 being simulated.
        sampling_function_name (str): Name of the sampling function in
            btk/utils.py that determines how objects are drawn from the catalog
            to create blends.
        verbose (bool): If True prints description at multiple steps.

    Returns:
        Generator objects that draws the blend scene.
    """
    if sampling_function_name != "None":
        try:
            utils_filename = user_config_dict['utils_filename']
            spec = importlib.util.spec_from_file_location(
                'blend_utils', utils_filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules['blend_utils'] = module
            spec.loader.exec_module(module)
            sampling_function = getattr(module, sampling_function_name)
        except(AttributeError) as e:
            print(e)
            utils_filename = os.path.join(
                os.path.dirname(btk.__file__), 'utils.py')
            spec = importlib.util.spec_from_file_location(
                'blend_utils', utils_filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules['blend_utils'] = module
            spec.loader.exec_module(module)
            sampling_function = getattr(module, sampling_function_name)
    else:
        utils_filename = os.path.join(
            os.path.dirname(btk.__file__), 'utils.py')
        sampling_function = None
    blend_generator = btk.create_blend_generator.generate(param, catalog,
                                                          sampling_function)
    if verbose:
        print(f"Blend generator draws from {param.catalog_name} catalog "
              f"with {sampling_function_name} sampling function defined "
              f" in {utils_filename}")
    return blend_generator


# REVIEW:
#  * Removed unused argument in docstring.
#  * Specified that param is a config.Simulation_params object, correct?
def get_obs_generator(param, user_config_dict, observe_function_name, verbose):
    """Returns generator object that generates class describing the observing
    conditions.

    Args:
        param (:obj:`config.Simulation_params`): Parameter values for btk simulations.
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        observe_function_name (str): Name of the function in btk/utils.py to
            set the observing conditions under which the galaxies are drawn.
        verbose (bool): If True prints description at multiple steps.

    Returns:
        Generator objects that generates class with the observing conditions.
    """
    if observe_function_name != "None":
        # search for obs function in user input utils, else in btk.utils
        try:
            utils_filename = user_config_dict['utils_filename']
            spec = importlib.util.spec_from_file_location(
                'obs_utils', utils_filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules['obs_utils'] = module
            spec.loader.exec_module(module)
            observe_function = getattr(module, observe_function_name)
        except AttributeError as e:
            print(e)
            utils_filename = os.path.join(
                os.path.dirname(btk.__file__), 'utils.py')
            spec = importlib.util.spec_from_file_location(
                'obs_utils', utils_filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules['obs_utils'] = module
            spec.loader.exec_module(module)
            observe_function = getattr(module, observe_function_name)
    else:
        observe_function = None
    observing_generator = btk.create_observing_generator.generate(
        param, observe_function)
    if verbose:
        print(f"Observing conditions generated using {observe_function_name}"
              " function defined in {utils_filename}")
    return observing_generator


# REVIEW:
#  * Specified that param is a config.Simulation_params object, correct?
#  * generator typo
def make_draw_generator(param, user_config_dict, simulation_config_dict,
                        multiprocess=False, cpus=1):
    """Returns a generator that yields simulations of blend scenes.

    Args:
        param (:obj:`config.Simulation_params`): Parameter values for btk simulations.
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        simulation_config_dict (dict): Dictionary which sets the parameter
            values of simulations of the blend scene.
        multiprocess: If true, performs multiprocessing of measurement.
        cpus: If multiprocessing is True, then number of parallel processes to
             run on [Default :1].

    Returns:
        Generator objects that generates output of blend scene.

    """
    # Load catalog to simulate objects from
    catalog = get_catalog(
        param, user_config_dict,
        str(simulation_config_dict['selection_function']), param.verbose)
    # Generate catalogs of blended objects
    blend_generator = get_blend_generator(
        param, user_config_dict, catalog,
        str(simulation_config_dict['sampling_function']),
        param.verbose)
    # Generate observing conditions
    observing_generator = get_obs_generator(
        param, user_config_dict,
        str(simulation_config_dict['observe_function']),
        param.verbose)
    if multiprocess:
        print(f"Multiprocess draw over {cpus} cpus")
    # Generate images of blends in all the observing bands
    draw_blend_generator = btk.draw_blends.generate(
        param, blend_generator, observing_generator,
        multiprocessing=multiprocess, cpus=cpus)
    return draw_blend_generator


def get_measurement_class(user_config_dict, verbose):
    """Returns the class that when input to btk.measure yields the output from
    the measurement algorithm.

    If utils_input.measure_function is input in user_config_dict then a class
    with that name is loaded from utils_filename to generate the
    btk.measure.Measurement_params class. If measure_function is 'None', then
    default_measure_function from btk.utils is returned as measurement class.
    The measurement class determines how objects in the blend images are
    detected/deblended/measured.

    Args:
        user_config_dict: Dict with information to run user defined functions.
        verbose (bool): If True prints description at multiple steps.

    Returns:
        Class derived from btk.measure.Measurement_params that describes
        how the detection/deblending/measurement algorithm processes the blend
        scene image.
    """
    measure_class_name = user_config_dict['utils_input']['measure_function']
    if measure_class_name == 'None':
        measure_class_name = 'Basic_measure_params'
        utils_filename = os.path.join(os.path.dirname(btk.__file__),
                                      'utils.py')
    else:
        utils_filename = user_config_dict['utils_filename']
    spec = importlib.util.spec_from_file_location('meas_utils',
                                                  utils_filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules['meas_utils'] = module
    spec.loader.exec_module(module)
    if verbose:
        print(f"Measurement class set as {measure_class_name} defined in "
              f"{utils_filename}")
    return getattr(module, measure_class_name)


# REVIEW:
#  * Specified that param is a config.Simulation_params object, correct?
#  * Is this the correct way of specifying a class object in a docstring?
def make_measure_generator(param, user_config_dict, draw_blend_generator,
                           multiprocess=False, cpus=1):
    """Returns a generator that yields simulations of blend scenes.

    Args:
        param (:obj:`config.Simulation_params`): Parameter values for btk simulations.
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        draw_blend_generator : Generator that yields simulations of blend
            scenes.
        multiprocess: If true, performs multiprocessing of measurement.
        cpus: If multiprocessing is True, then number of parallel processes to
             run on [Default :1].

    Returns:
        Generator objects that yields measured values by the measurement
        algorithm over the batch.

    """
    # get class that describes how measurement algorithm performs measurement
    measure_class = get_measurement_class(user_config_dict,
                                          param.verbose)
    if multiprocess:
        print(f"Multiprocess measurement over {cpus} cpus")
    # get generator that yields measured values.
    measure_generator = btk.measure.generate(
        measure_class(), draw_blend_generator, param,
        multiprocessing=multiprocess, cpus=cpus)
    return measure_generator


def get_metrics_class(user_config_dict, verbose):
    """Returns the class that when input to btk.compute_metrics yields the
    output from the metrics computed for measurement algorithm.

    If utils_input.metrics_function is input in user_config_dict then a class
    with that name is loaded from utils_filename to generate the
    btk.measure.Measurement_params class. If metrics_function is 'None', then
    default_metrics_function from btk.utils is returned as measurement class.
    The metrics class determines how detected/deblended/measured output
    performance can be assessed.

    Args:
        user_config_dict: Dict with information to run user defined functions.
        verbose (bool): If True prints description at multiple steps.

    Returns:
        Class derived from btk.measure.Measurement_params that describes
        how the detection/deblending/measurement algorithm processes the blend
        scene image.
    """
    metrics_class_name = user_config_dict['utils_input']['metrics_function']
    if metrics_class_name == 'None':
        metrics_class_name = 'Basic_metric_params'
        utils_filename = os.path.join(os.path.dirname(btk.__file__),
                                      'utils.py')
    else:
        utils_filename = user_config_dict['utils_filename']
    spec = importlib.util.spec_from_file_location('metrics_utils',
                                                  utils_filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules['metrics_utils'] = module
    spec.loader.exec_module(module)
    if verbose:
        print(f"Measurement class set as {metrics_class_name} defined in "
              f"{utils_filename}")
    return getattr(module, metrics_class_name)


# REVIEW:
#  Fixed a few typos.
def get_output_path(user_config_dict, verbose):
    """Returns path where btk output will be stored to disk.

    If output folder does not exist it will be created

    Args:
        user_config_dict: Dict with information to run user defined functions
            and store results.
        verbose (bool): If True prints description at multiple steps.

    Returns:
        string with the path to where output must be stored to disk.
    """
    output_dir = user_config_dict['output_dir']
    output_name = user_config_dict['output_name']
    if not os.path.isdir(output_dir):
        subprocess.call(['mkdir', output_dir])
        if verbose:
            print(f"Output directory created at {output_dir}")
    output_path = os.path.join(output_dir, output_name)
    if not os.path.isdir(output_path):
        subprocess.call(['mkdir', output_path])
        if verbose:
            print(f"Test output directory created at {output_path}")
    if verbose:
        print(f"Output will be saved at {output_path}")
    return output_path


# REVIEW:
#  Typo in output_path
def save_config_file(param, user_config_dict, simulation_config_dict,
                     simulation, output_path):
    """Saves all parameter values to a yaml file and writes it to disk.

    Args:
        param(class): Parameter values for btk simulations.
        user_config_dict: Dict with information to run user defined functions
            and store results.
        simulation_config_dict (dict): Dictionary which sets the parameter
            values of simulations of the blend scene.
        simulation: Name of simulation to run btk test on.
        output_path(string): Path to where output must be stored to disk.

    """
    save_config_dict = {'simulation': simulation}
    # save btk.Simulation_param values used.
    save_config_dict.update({'btk_Simulation_params': param.__dict__})
    # save simulation values from input config file.
    save_config_dict.update({'simulation_config': simulation_config_dict})
    # save user defined function and file names.
    save_config_dict.update({'user_config': user_config_dict})
    output_name = os.path.join(output_path, simulation + '_config.yaml')
    with open(output_name, 'w') as outfile:
        yaml.dump(save_config_dict, outfile)
    print("Configuration file saved at", output_name)


def main(args):
    """
    Runs btk test on simulation parameters and algorithm specified in input
    yaml config file.

    Args:
        args: Class with parameters controlling how btk should be run.
    """
    config_dict = read_configfile(args.configfile, args.simulation,
                                  args.verbose)
    for i, s in enumerate(config_dict['simulation']):
        simulation_config_dict = config_dict['simulation'][s]
        user_config_dict = config_dict['user_input']
        catalog_name = os.path.join(
            user_config_dict['data_dir'],
            simulation_config_dict['catalog'])
        # Set parameter values in param
        param = get_config_class(simulation_config_dict,
                                 catalog_name, args.verbose)
        # Set seed
        np.random.seed(int(param.seed))
        if args.multiprocess:
            if args.cpus is None:
                cpus = multiprocessing.cpu_count()
            else:
                cpus = args.cpus
        else:
            cpus = 1
        # Generate images of blends in all the observing bands
        draw_blend_generator = make_draw_generator(
            param, user_config_dict, simulation_config_dict,
            args.multiprocess, cpus=cpus)
        # Create generator for measurement algorithm outputs
        measure_generator = make_measure_generator(param, user_config_dict,
                                                   draw_blend_generator,
                                                   args.multiprocess,
                                                   cpus=cpus)
        # get metrics class that can generate metrics
        metrics_class = get_metrics_class(user_config_dict,
                                          param.verbose)
        test_size = int(simulation_config_dict['test_size'])
        metrics_param = metrics_class(measure_generator, param)
        output_path = get_output_path(user_config_dict, param.verbose)
        output_name = os.path.join(output_path, s + '_metrics_results.dill')
        results = btk.compute_metrics.run(metrics_param, test_size=test_size)
        with open(output_name, 'wb') as handle:
            dill.dump(results, handle)
        print("BTK outputs saved at ", output_name)
        save_config_file(param, user_config_dict, simulation_config_dict,
                         s, output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', default='two_gal',
                        choices=['two_gal', 'multi_gal', 'group', 'all'],
                        help='Name of the simulation to use, taken from '
                             'your configuration file [Default:"two_gal"]')
    parser.add_argument('--configfile', default='input/btk-config.yaml',
                        help='Configuration file containing a set of option '
                             'values. The content of this file will be overwritten '
                             'by any given command line options.'
                             "[Default:'input/btk-config.yaml']")
    parser.add_argument('--multiprocess', action='store_true',
                        help='If True multiprocess is performed for '
                             'measurement in the batch')
    parser.add_argument('--cpus', nargs='?', const=1, type=int,
                        help='Number of cpus. Must be int or None [Default:1]')
    parser.add_argument('--verbose', action='store_true',
                        help='If True prints description at multiple steps')
    # REVIEW:
    #  avoid shadowing name in function above.
    Args = parser.parse_args()
    main(Args)
