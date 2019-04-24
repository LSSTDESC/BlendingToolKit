import pytest
import subprocess
import imp
import os
import sys
import numpy as np

# TODO
# check output dump file
# delete test output at end
# test multiprocessing


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


@pytest.mark.timeout(45)
def test_input():
    sys.path.append("..")
    """
    tests if btk_input script is correctly executed
    """
    command = ['python', 'btk_input.py', '--name', 'unit_test',
               '--configfile', 'tests/test-config.yaml']
    for s in ['all', 'two_gal', 'multi_gal', 'group']:
        subprocess.check_call(command + ['--simulation', s])


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
    try:
        stack_meas(param, user_config_dict, simulation_config_dict, btk_input)
    except ImportError:
        print("stack not found")
    try:
        scarlet_meas(param, user_config_dict,
                     simulation_config_dict, btk_input)
    except ImportError:
        print("scarlet not found")
    pass
