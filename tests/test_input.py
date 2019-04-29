import pytest
import subprocess
import imp
import os
import sys
import numpy as np
import astropy
import dill

# TODO
# check output dump file
# test multiprocessing


@pytest.mark.timeout(5)
def test_parse_config():
    """Checks if input config files are parsed correctly"""
    simulations = ['all', 'two_gal', 'multi_gal', 'group']
    for simulation in simulations:
        args = Input_Args(simulation=simulation)
        sys.path.append(os.getcwd())
        btk_input = __import__('btk_input')
        config_dict = btk_input.read_configfile(
            args.configfile, args.simulation,  args.verbose)
        assert set(config_dict.keys()) == set(['user_input', 'simulation']), \
            "config_dict must have only 'user_input', 'simulation'. Found" \
            f"{config_dict.keys()}"
        set_keys = set(config_dict['simulation'].keys())
        if simulation == 'all':
            assert set(simulations[1:]) == set_keys, "Invalid simulation keys"\
                f" .Expected {simulations[1:]} found {set_keys}"
        else:
            assert set([simulation]) == set_keys, "Invalid "\
                f"simulation name. Expected {simulation}. Got {set_keys}"
    pass


class Input_Args(object):
    def __init__(self, simulation='two_gal', name='unit_test',
                 configfile='tests/test-config.yaml', verbose=True):
        self.simulation = simulation
        self.configfile = configfile
        self.name = name
        self.verbose = verbose


@pytest.mark.timeout(5)
def test_input_draw():
    args = Input_Args()
    sys.path.append(os.getcwd())
    btk_input = __import__('btk_input')
    config_dict = btk_input.read_configfile(args.configfile, args.simulation,
                                            args.verbose)
    simulation_config_dict = config_dict['simulation'][args.simulation]
    user_config_dict = config_dict['user_input']
    catalog_name = os.path.join(
        user_config_dict['data_dir'],
        simulation_config_dict['catalog'])
    # Set parameter values in param
    param = btk_input.get_config_class(simulation_config_dict,
                                       catalog_name, args.verbose)
    # Set seed
    np.random.seed(int(param.seed))
    draw_blend_generator = btk_input.make_draw_generator(
            param, user_config_dict, simulation_config_dict)
    draw_output = next(draw_blend_generator)
    assert len(draw_output['blend_list']) == 8, "Default batch should return 8"
    assert len(draw_output['blend_list'][3]) < 3, "Default max_number should \
        generate 2 or 1 galaxies per blend."
    assert draw_output['obs_condition'][5][0].survey_name == 'LSST', "Default \
        observing survey is LSST."
    test_draw = imp.load_source("", 'tests/test_draw.py')
    test_draw.match_blend_images_default(draw_output['blend_images'])
    test_draw.match_isolated_images_default(draw_output['isolated_images'])
    pass


def check_output_file(user_config_dict, simulation):
    """Check if metrics output is correctly written to file"""
    ouput_path = os.path.join(user_config_dict['output_dir'],
                              user_config_dict['output_name'])
    if not os.path.isdir(ouput_path):
        raise FileNotFoundError(f"btk output must be saved at {ouput_path}")
    output_name = os.path.join(ouput_path,
                               simulation + '_metrics_results.dill')
    if not os.path.isfile(output_name):
        raise FileNotFoundError(f"btk output must be saved at {output_name}")
    pass


def check_output_values(user_config_dict, simulation):
    """Check if metrics output is correctly written to file"""
    ouput_path = os.path.join(user_config_dict['output_dir'],
                              user_config_dict['output_name'])
    output_name = os.path.join(ouput_path,
                               simulation + '_metrics_results.dill')
    with open(output_name, 'rb') as handle:
        results = dill.load(handle)
    result_keys = ['detection', 'segmentation', 'flux', 'shapes']
    assert set(results.keys()) == set(result_keys), "Results have incorrect"\
        f"keys. Found {results.keys()}, expected {result_keys}"
    if not isinstance(results['detection'][0], astropy.table.Table):
        raise ValueError("Expected astropy table in results['detection'][0],  "
                         f"got {type(results['detection'][0])} ")
    if not isinstance(results['detection'][1], astropy.table.Table):
        raise ValueError("Expected astropy table in results['detection'][1],  "
                         f"got {type(results['detection'][1])} ")
    if not isinstance(results['detection'][2], list):
        raise ValueError("Expected astropy table in results['detection'][2],  "
                         f"got {type(results['detection'][2])} ")
    pass


def delete_output_file(user_config_dict, simulation):
    """Delete all files written to disk"""
    ouput_path = os.path.join(user_config_dict['output_dir'],
                              user_config_dict['output_name'])
    output_name = os.path.join(ouput_path,
                               simulation + '_metrics_results.dill')
    yaml_output_name = os.path.join(ouput_path,
                                    simulation + '_param.yaml')
    if os.path.isfile(yaml_output_name):
        subprocess.call(['rm', yaml_output_name])
    if os.path.isfile(output_name):
        subprocess.call(['rm', output_name])
    if os.path.isdir(ouput_path):
        subprocess.call(['rmdir', ouput_path])
    return


@pytest.mark.timeout(45)
def test_input_output():
    """Checks output of btk called in test_input for input simulation.

    Checks that the output files have correct values and are saved in correct
    format and location.

    """
    for simulation in ['two_gal', 'multi_gal', 'group']:
        command = ['python', 'btk_input.py', '--name', 'unit_test',
                   '--configfile', 'tests/test-config.yaml']
        subprocess.call(command + ['--simulation', simulation])
        args = Input_Args(simulation=simulation)
        sys.path.append(os.getcwd())
        btk_input = __import__('btk_input')
        config_dict = btk_input.read_configfile(
            args.configfile, args.simulation, args.verbose)
        user_config_dict = config_dict['user_input']
        try:
            check_output_file(user_config_dict, simulation)
            check_output_values(user_config_dict, simulation)
            print("btk output files created")
        except FileNotFoundError as e:
            print("btk output files not found")
            raise e
        except ValueError as e:
            print("btk output files were not in correct format")
            raise e
        finally:
            delete_output_file(user_config_dict, simulation)
            print("deleted files")
        pass


def basic_meas(param, user_config_dict, simulation_config_dict, btk_input):
    # Set seed
    np.random.seed(int(param.seed))
    draw_blend_generator = btk_input.make_draw_generator(
            param, user_config_dict, simulation_config_dict)
    measure_generator = btk_input.make_measure_generator(
            param, user_config_dict, draw_blend_generator)
    test_detect_centers = [[[57, 67], [63, 59], [49, 57]],
                           [[49, 67]],
                           [[59, 67]],
                           [[54, 63], [60, 49]],
                           ]
    output, deb, _ = next(measure_generator)
    for i in range(len(output['blend_list'])):
        detected_centers = deb[i]['peaks']
        np.testing.assert_array_almost_equal(
            detected_centers, test_detect_centers[i], decimal=3,
            err_msg="Did not get desired detected_centers")
    pass


def sep_meas(param, user_config_dict, simulation_config_dict, btk_input):
    # Set seed
    np.random.seed(int(param.seed))
    draw_blend_generator = btk_input.make_draw_generator(
            param, user_config_dict, simulation_config_dict)
    user_config_dict['utils_input']['measure_function'] = 'SEP_params'
    measure_generator = btk_input.make_measure_generator(
            param, user_config_dict, draw_blend_generator)
    test_detect_centers = [[[62.633432, 58.845595], [56.823835, 66.857761],
                            [55.138673, 59.557461], [49.164199, 56.989852]],
                           [[49.245587, 67.135604]],
                           [[58.834318, 66.726691]],
                           [[68.389972, 61.936116], [60.043505, 47.8969331],
                            [54.581039, 61.718651]]
                           ]
    output, deb, _ = next(measure_generator)
    for i in range(len(output['blend_list'])):
        detected_centers = deb[i]['peaks']
        np.testing.assert_array_almost_equal(
            detected_centers, test_detect_centers[i], decimal=3,
            err_msg="Did not get desired detected_centers")
    pass


def stack_meas(param, user_config_dict, simulation_config_dict, btk_input):
    # Set seed
    np.random.seed(int(param.seed))
    draw_blend_generator = btk_input.make_draw_generator(
            param, user_config_dict, simulation_config_dict)
    user_config_dict['utils_input']['measure_function'] = 'Stack_params'
    measure_generator = btk_input.make_measure_generator(
            param, user_config_dict, draw_blend_generator)
    test_detect_centers = [[[58.063703, 59.749699], [61.157868, 69.30290],
                            [68.304245, 61.537312]],
                           [[59.915507, 50.167592], [65.766700, 65.105297]],
                           [[51.243380, 58.382503], [54.900160, 68.5794316]],
                           [[70.645195, 51.627339], [63.226545, 56.1251558]]
                           ]
    output, deb, meas = next(measure_generator)
    for i in range(len(output['blend_list'])):
        detected_centers = deb[i][1]
        np.testing.assert_array_almost_equal(
            detected_centers, test_detect_centers[i], decimal=3,
            err_msg="Did not get desired detected_centers")
    pass


def scarlet_meas(param, user_config_dict, simulation_config_dict, btk_input):
    # Set seed
    np.random.seed(int(param.seed))
    draw_blend_generator = btk_input.make_draw_generator(
            param, user_config_dict, simulation_config_dict)
    user_config_dict['utils_input']['measure_function'] = 'Scarlet_params'
    measure_generator = btk_input.make_measure_generator(
            param, user_config_dict, draw_blend_generator)
    test_detect_centers = [[[58.063703, 59.749699], [61.157868, 69.30290],
                            [68.304245, 61.537312]],
                           [[59.915507, 50.167592], [65.766700, 65.105297]],
                           [[51.243380, 58.382503], [54.900160, 68.5794316]],
                           [[70.645195, 51.627339], [63.226545, 56.1251558]]
                           ]
    output, deb, _ = next(measure_generator)
    for i in range(len(output['blend_list'])):
        detected_centers = deb[i]['peaks']
        np.testing.assert_array_almost_equal(
            detected_centers, test_detect_centers[i], decimal=3,
            err_msg="Did not get desired detected_centers")
    pass


@pytest.mark.timeout(25)
def test_measure():
    args = Input_Args()
    sys.path.append(os.getcwd())
    btk_input = __import__('btk_input')
    config_dict = btk_input.read_configfile(args.configfile, args.simulation,
                                            args.verbose)
    simulation_config_dict = config_dict['simulation'][args.simulation]
    simulation_config_dict['max_number'] = 6
    simulation_config_dict['batch_size'] = 4
    user_config_dict = config_dict['user_input']
    catalog_name = os.path.join(
        user_config_dict['data_dir'],
        simulation_config_dict['catalog'])
    # Set parameter values in param
    param = btk_input.get_config_class(simulation_config_dict,
                                       catalog_name, args.verbose)
    basic_meas(param, user_config_dict, simulation_config_dict, btk_input)
    try:
        sep_meas(param, user_config_dict, simulation_config_dict, btk_input)
    except ImportError:
        print("sep not found")
    #try:
        #stack_meas(param, user_config_dict, simulation_config_dict, btk_input)
    #except ImportError:
    #    print("stack not found")
    #try:
    #    scarlet_meas(param, user_config_dict,
    #                simulation_config_dict, btk_input)
    except ImportError:
        print("scarlet not found")
    pass


def basic_metric_two_gal(output_name):
    """Loads metric results dill file and compares it to target value"""
    with open(output_name, 'rb') as handle:
        results = dill.load(handle)
    detected_metrics = np.array(results['detection'][2])
    test_metric_summary = np.array(
        [[1, 1, 0, 0, 0], [2, 2, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0],
         [2, 2, 0, 0, 0], [2, 2, 0, 0, 0], [2, 2, 0, 0, 0], [1, 1, 0, 0, 0],
         [2, 1, 1, 0, 0], [2, 2, 0, 0, 0], [1, 1, 0, 0, 0], [2, 2, 0, 0, 0],
         [2, 1, 1, 0, 0], [2, 1, 1, 0, 0], [2, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    np.testing.assert_array_almost_equal(
        detected_metrics, test_metric_summary, decimal=3,
        err_msg="Did not get desired detection metrics summary")
    pass


@pytest.mark.timeout(15)
def test_metrics():
    simulations = ['two_gal', ]
    for simulation in simulations:
        command = ['python', 'btk_input.py', '--name', 'unit_test',
                   '--configfile', 'tests/test-config.yaml']
        subprocess.call(command + ['--simulation', simulation])
        args = Input_Args(simulation=simulation)
        sys.path.append(os.getcwd())
        btk_input = __import__('btk_input')
        config_dict = btk_input.read_configfile(
            args.configfile, args.simulation, args.verbose)
        user_config_dict = config_dict['user_input']
        ouput_path = os.path.join(user_config_dict['output_dir'],
                                  user_config_dict['output_name'])
        output_name = os.path.join(ouput_path,
                                   simulation + '_metrics_results.dill')
        try:
            basic_metric_two_gal(output_name)
        except FileNotFoundError as e:
            print("btk output files not found")
            raise e
        except ValueError as e:
            print("btk output files were not in correct format")
            raise e
        finally:
            delete_output_file(user_config_dict, simulation)
            print("deleted files")
        pass
