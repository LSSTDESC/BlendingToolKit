import yaml
import types


def parse_config(config_gen, simulation, verbose):
    """Parses the config generator with the config yaml file information.

    Args:
        config_gen: Generator with values i the yaml config file.
        simulation: Name of simulation to run btk test on.

    Returns:
        dictionary with basic user input and parameter values to generate
        simulations set by args.simulation
    """
    config_dict = {'user_input': None, 'simulation': {}}
    for doc in config_gen:
        if 'user_input' in doc.keys():
            config_dict['user_input'] = doc['user_input']
            continue
        if 'simulation' in doc.keys():
            if (simulation == doc['simulation'] or simulation == 'all'):
                config_dict['simulation'][doc['simulation']] = doc['config']
                if verbose:
                    print(f"{doc['simulation']} parameter values loaded to "
                          "config dict")
    if not config_dict['simulation']:
        raise ValueError('simulation name does not exist in config yaml file')
    print(config_dict)
    return config_dict


def read_configfile(filename, simulation, verbose):
    """Reads config YAML file.

    Args:
        filename: YAML config file with configuration values to run btk.
        simulation: Name of simulation to run btk test on.

    Returns:
        dictionary with btk configuration.

    """
    with open(filename, 'r') as stream:
        config_gen = yaml.safe_load_all(stream)  # load all yaml files here
        if verbose:
            print(f"config file {filename} loaded", config_gen)
        if not isinstance(config_gen, types.GeneratorType):
            raise ValueError("input config yaml file not should contain at least "
                             "two documents, one for user input, other denoting"
                             "simulation parameters")
        config_dict = parse_config(config_gen, simulation, verbose)
    return config_dict


def main(args):
    """
    Runs btk test on simulation parameters and algorithm specified in input
    yaml config file.

    Args:
        args: Class with parameters controlling how btk should be run.
    """
    config_dict = read_configfile(args.configfile, args.simulation,
                                  args.verbose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', default='two_gal',
                        choices=['two_gal', 'multi_gal', 'group', 'all'],
                        help='Name of the simulation to use, taken from '
                        'your configuration file (btk-config.yaml). '
                        "[Default:'two_gal']")
    parser.add_argument('--configfile', default='btk-config.yaml',
                        help='Configuration file containing a set of option '
                        'values. The content of this file will be overwritten '
                        'by any given command line options.'
                        "[Default:'btk-config.yaml']")
    parser.add_argument('--name', default='test_1',
                        help='Name of the btk run. Output will be stored in '
                        'a directory under this name [Default: "test1:].')
    parser.add_argument('--verbose', action='store_true',
                        help='If True prints returns description at multiple '
                        'steps')
    args = parser.parse_args()
    main(args)
