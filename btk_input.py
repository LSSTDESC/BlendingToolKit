import yaml
import types
import btk
import os
import numpy as np
import imp
import dill
import subprocess


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
            if (simulation == doc['simulation'] or simulation == 'all'):
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


def get_catalog(param, selection_function_name, verbose):
    """Returns catalog from which objects are simulated by btk

    Args:
        param: Class with btk simulation parameters.
        selection_function_name (str): Name of the selection function in
            btk/utils.py.
        verbose (bool): If True prints description at multiple steps.

    Returns:
        `astropy.table.Table` with parameters corresponding to objects being
        simulated.
    """
    if selection_function_name != "None":
        btk_utils = os.path.join(os.path.dirname(btk.__file__), 'utils.py')
        utils = imp.load_source("", btk_utils)
        selection_function = getattr(utils, selection_function_name)
    else:
        selection_function = None
    catalog = btk.get_input_catalog.load_catalog(
        param, selection_function=selection_function)
    if verbose:
        print(f"Loaded {param.catalog_name} catalog with "
              f"{selection_function_name} selection "
              "function defined in btk/utils.py")
    return catalog


def get_blend_generator(param, catalog, sampling_function_name, verbose):
    """Returns generator object that generates catalog describing blended
    objects.

    Args:
        param (class): Parameter values for btk simulations.
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
        btk_utils = os.path.join(os.path.dirname(btk.__file__), 'utils.py')
        utils = imp.load_source("", btk_utils)
        sampling_function = getattr(utils, sampling_function_name)
    else:
        sampling_function = None
    blend_generator = btk.create_blend_generator.generate(param, catalog,
                                                          sampling_function)
    if verbose:
        print(f"Blend generator draws from {param.catalog_name} catalog "
              f"with {sampling_function_name} sampling function defined "
              " in btk/utils.py")
    return blend_generator


def get_obs_generator(param, observe_function_name, verbose):
    """Returns generator object that generates class describing the observing
    conditions.

    Args:
        param (class): Parameter values for btk simulations.
        catalog: `astropy.table.Table` with parameters corresponding to objects
                 being simulated.
        observe_function_name (str): Name of the function in btk/utils.py to
            set the observing conditions under which the galaxies are drawn.
        verbose (bool): If True prints description at multiple steps.

    Returns:
        Generator objects that generates class with the observing conditions.
    """
    if observe_function_name != "None":
        btk_utils = os.path.join(os.path.dirname(btk.__file__), 'utils.py')
        utils = imp.load_source("", btk_utils)
        observe_function = getattr(utils, observe_function_name)
    else:
        observe_function = None
    observing_generator = btk.create_observing_generator.generate(
        param, observe_function)
    if verbose:
        print(f"Observing conditions generated using {observe_function_name}"
              " function defined in btk/utils.py")
    return observing_generator


def make_draw_generator(param, user_config_dict, simulation_config_dict):
    """Returns a generator that yields simulations of blend scenes.

    Args:
        param (class): Parameter values for btk simulations.
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        simulation_config_dict (dict): Dictionary which sets the parameter
            values of simulations of the blend scene.

    Returns:
        Generator objects that generates output of blend scene.

    """
    # Load catalog to simulate objects from
    catalog = get_catalog(
        param, str(simulation_config_dict['selection_function']),
        param.verbose)
    # Generate catalogs of blended objects
    blend_genrator = get_blend_generator(
        param, catalog, str(simulation_config_dict['sampling_function']),
        param.verbose)
    # Generate observing conditions
    observing_genrator = get_obs_generator(
        param, str(simulation_config_dict['observe_function']),
        param.verbose)
    # Generate images of blends in all the observing bands
    draw_blend_generator = btk.draw_blends.generate(param, blend_genrator,
                                                    observing_genrator)
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
    utils = imp.load_source("", utils_filename)
    measure_class = getattr(utils, measure_class_name)
    if verbose:
        print(f"Measurement class set as {measure_class_name} defined in "
              f"{utils_filename}")
    return measure_class


def make_measure_generator(param, user_config_dict,
                           draw_blend_generator):
    """Returns a generator that yields simulations of blend scenes.

    Args:
        param (class): Parameter values for btk simulations.
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        draw_blend_generator : Generator that yields simulations of blend
            scenes.

    Returns:
        Generator objects that yields measured values by the measurement
        algorithm over the batch.

    """
    # get class that describes how measurement algorithm performs measurement
    measure_class = get_measurement_class(user_config_dict,
                                          param.verbose)
    # get generator that yields measured values.
    measure_generator = btk.measure.generate(
            measure_class(), draw_blend_generator, param)
    return measure_generator


def get_metrics_class(user_config_dict, verbose):
    """Returns the class that when input to btk.compute_metrics yields the
    output from the metrics computed for measurement algorithm.

    If utils_input.metrics_function is input in user_config_dict then a class
    with that name is loaded from utils_filename to generate the
    btk.measure.Measurement_params class. If metrics_function is 'None', then
    default_metrics_function from btk.utils is returned as measurement class.
    The metrics class determines how detected/deblended/measured output
    performance can be assesed.

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
    utils = imp.load_source("", utils_filename)
    metrics_class = getattr(utils, metrics_class_name)
    if verbose:
        print(f"Measurement class set as {metrics_class_name} defined in "
              f"{utils_filename}")
    return metrics_class


def get_ouput_path(user_config_dict, verbose):
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
    ouput_path = os.path.join(output_dir, output_name)
    if not os.path.isdir(ouput_path):
        subprocess.call(['mkdir', ouput_path])
        if verbose:
            print(f"Test output directory created at {ouput_path}")
    if verbose:
        print(f"Output will be saved at {ouput_path}")
    return ouput_path


def save_config_file(param, user_config_dict, simulation_config_dict,
                     simulation, ouput_path, verbose):
    """Saves all parameter values to a yaml file and writes it to disk.

    Args:
        param(class): Parameter values for btk simulations.
        user_config_dict: Dict with information to run user defined functions
            and store results.
        simulation_config_dict (dict): Dictionary which sets the parameter
            values of simulations of the blend scene.
        simulation: Name of simulation to run btk test on.
        output_path(string): Path to where output must be stored to disk.
        verbose (bool): If True prints description at multiple steps.

    """
    save_config_dict = {'simulation': simulation}
    # save btk.Simulation_param values used.
    save_config_dict.update({'btk_Simulation_params': param.__dict__})
    # save simulation values from input config file.
    save_config_dict.update({'simulation_config': simulation_config_dict})
    # save user defined function and file names.
    save_config_dict.update({'user_config': user_config_dict})
    output_name = os.path.join(ouput_path, simulation + '_config.yaml')
    with open(output_name, 'w') as outfile:
        yaml.dump(save_config_dict, outfile)


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
        # Generate images of blends in all the observing bands
        draw_blend_generator = make_draw_generator(
            param, user_config_dict, simulation_config_dict)
        # Create generator for measurement algorithm outputs
        measure_generator = make_measure_generator(param, user_config_dict,
                                                   draw_blend_generator)
        # get metrics class that can generate metrics
        metrics_class = get_metrics_class(user_config_dict,
                                          param.verbose)
        test_size = int(simulation_config_dict['test_size'])
        metrics_param = metrics_class(measure_generator, param)
        ouput_path = get_ouput_path(user_config_dict, param.verbose)
        output_name = os.path.join(ouput_path, s + '_metrics_results.dill')
        results = btk.compute_metrics.run(metrics_param, test_size=test_size)
        with open(output_name, 'wb') as handle:
            dill.dump(results, handle)
        save_config_file(param, user_config_dict, simulation_config_dict,
                         s, ouput_path, param.verbose)


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
    parser.add_argument('--name', default='test_1',
                        help='Name of the btk run. Output will be stored in '
                        'a directory under this name [Default: "test1"].')
    parser.add_argument('--verbose', action='store_true',
                        help='If True prints description at multiple steps')
    args = parser.parse_args()
    main(args)
