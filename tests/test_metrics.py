import pytest
import btk
import imp
import os
import sys
import numpy as np


def compare_basic_metric(param, user_config_dict,
                         simulation_config_dict, btk_input):
    # Set seed
    test_metric_summary = np.array(
        [[5, 3, 2, 0, 0, 3, 2, 0, 0],
         [2, 1, 1, 0, 0, 1, 1, 0, 0],
         [1, 1, 0, 0, 0, 1, 0, 0, 0],
         [6, 2, 4, 0, 0, 2, 4, 0, 0]])
    np.random.seed(int(param.seed))
    draw_blend_generator = btk_input.make_draw_generator(
            param, user_config_dict, simulation_config_dict)
    measure_generator = btk_input.make_measure_generator(
            param, user_config_dict, draw_blend_generator)
    metric_param = btk.utils.Basic_metric_params(
        meas_generator=measure_generator, param=param)
    results = btk.compute_metrics.run(metric_param, test_size=1)
    detected_metrics_summary = results['detection'][2]
    np.testing.assert_array_almost_equal(
        detected_metrics_summary, test_metric_summary, decimal=3,
        err_msg="Did not get desired detection metrics summary")
    pass


def run_metrics_basic():
    test_input = imp.load_source("", 'tests/test_input.py')
    args = test_input.Input_Args()
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
    compare_basic_metric(param, user_config_dict,
                         simulation_config_dict, btk_input)
    pass


def compare_sep_group_metric(param, user_config_dict,
                             simulation_config_dict, btk_input):
    # Set seed
    test_metric_summary = np.array(
        [[3, 1, 2, 0, 0, 1, 2, 0, 0],
         [2, 1, 1, 0, 0, 1, 1, 0, 0],
         [3, 1, 2, 0, 0, 1, 2, 0, 0],
         [2, 2, 0, 0, 0, 2, 0, 0, 0],
         [2, 1, 1, 0, 0, 1, 1, 0, 0],
         [3, 1, 2, 0, 0, 1, 2, 0, 0],
         [2, 1, 1, 0, 0, 1, 1, 0, 0],
         [5, 2, 3, 0, 0, 2, 3, 0, 0],
         [2, 1, 1, 0, 0, 1, 1, 0, 0],
         [2, 1, 1, 0, 0, 1, 1, 0, 0],
         [3, 1, 2, 0, 0, 1, 2, 0, 0],
         [5, 2, 3, 0, 0, 2, 3, 0, 0],
         [5, 1, 4, 0, 0, 1, 4, 0, 0],
         [6, 3, 3, 0, 0, 3, 3, 0, 0],
         [2, 1, 1, 0, 0, 1, 1, 0, 0],
         [6, 3, 3, 0, 0, 3, 3, 0, 0]])
    np.random.seed(int(param.seed))
    draw_blend_generator = btk_input.make_draw_generator(
            param, user_config_dict, simulation_config_dict)
    measure_generator = btk_input.make_measure_generator(
            param, user_config_dict, draw_blend_generator)
    metric_param = btk.utils.Basic_metric_params(
        meas_generator=measure_generator, param=param)
    results = btk.compute_metrics.run(metric_param, test_size=2)
    detected_metrics_summary = results['detection'][2]
    np.testing.assert_array_almost_equal(
        detected_metrics_summary, test_metric_summary, decimal=3,
        err_msg="Did not get desired detection metrics summary")
    pass


def run_metrics_sep():
    test_input = imp.load_source("", 'tests/test_input.py')
    simulations = ['group', ]
    for simulation in simulations:
        args = test_input.Input_Args(simulation=simulation)
        sys.path.append(os.getcwd())
        btk_input = __import__('btk_input')
        config_dict = btk_input.read_configfile(args.configfile, args.simulation,
                                                args.verbose)
        simulation_config_dict = config_dict['simulation'][args.simulation]
        user_config_dict = config_dict['user_input']
        user_config_dict['utils_input']['measure_function'] = 'SEP_params'
        catalog_name = os.path.join(
            user_config_dict['data_dir'],
            simulation_config_dict['catalog'])
        # Set parameter values in param
        param = btk_input.get_config_class(simulation_config_dict,
                                           catalog_name, args.verbose)
        compare_sep_group_metric(param, user_config_dict,
                                 simulation_config_dict, btk_input)
    pass


def compare_stack_group_metric(param, user_config_dict,
                               simulation_config_dict, btk_input):
    # Set seed
    test_metric_summary = np.array(
        [[3, 1, 2, 0, 0], [2, 1, 1, 0, 0], [3, 1, 2, 0, 0], [2, 2, 0, 0, 0],
         [2, 1, 1, 0, 0], [3, 1, 2, 0, 0], [2, 1, 1, 0, 0], [5, 0, 5, 0, 0],
         [2, 1, 1, 0, 0], [2, 1, 1, 0, 0], [3, 1, 2, 0, 0], [5, 2, 3, 0, 0],
         [5, 0, 5, 1, 0], [6, 3, 3, 0, 0], [2, 1, 1, 0, 0], [6, 3, 3, 0, 0]])
    np.random.seed(int(param.seed))
    draw_blend_generator = btk_input.make_draw_generator(
            param, user_config_dict, simulation_config_dict)
    measure_generator = btk_input.make_measure_generator(
            param, user_config_dict, draw_blend_generator)
    metric_param = btk.utils.Stack_metric_params(
        meas_generator=measure_generator, param=param)
    results = btk.compute_metrics.run(metric_param, test_size=2)
    detected_metrics_summary = results['detection'][2]
    np.testing.assert_array_almost_equal(
        detected_metrics_summary, test_metric_summary, decimal=3,
        err_msg="Did not get desired detection metrics summary")
    pass


def run_metrics_stack():
    test_input = imp.load_source("", 'tests/test_input.py')
    simulations = ['group', ]
    for simulation in simulations:
        args = test_input.Input_Args(simulation=simulation)
        sys.path.append(os.getcwd())
        btk_input = __import__('btk_input')
        config_dict = btk_input.read_configfile(args.configfile, args.simulation,
                                                args.verbose)
        simulation_config_dict = config_dict['simulation'][args.simulation]
        user_config_dict = config_dict['user_input']
        user_config_dict['utils_input']['measure_function'] = 'Stack_params'
        catalog_name = os.path.join(
            user_config_dict['data_dir'],
            simulation_config_dict['catalog'])
        # Set parameter values in param
        param = btk_input.get_config_class(simulation_config_dict,
                                           catalog_name, args.verbose)
        compare_stack_group_metric(param, user_config_dict,
                                   simulation_config_dict, btk_input)
    pass


@pytest.mark.timeout(15)
def test_metrics_all():
    run_metrics_basic()
    try:
        run_metrics_sep()
    except ImportError:
            print("sep not found")
    try:
        run_metrics_stack()
    except ImportError:
            print("stack not found")


@pytest.mark.timeout(3)
def test_detection_eff_matrix():
    """Tests detection efficiency matrix computation in utils by inputting a
    summary table with 4 entries, with number of true sources between 1-4 and
    all detected and expecting matrix with
    secondary diagonal being one"""
    summary = np.array([[1, 1, 0, 0, 0], [2, 2, 0, 0, 0],
                        [3, 3, 0, 0, 0], [4, 4, 0, 0, 0]])
    num = 4
    eff_matrix = btk.utils.get_detection_eff_matrix(summary, num)
    test_eff_matrix = np.eye(num+2)[:, :num+1] * 100
    test_eff_matrix[0, 0] = 0.
    np.testing.assert_array_equal(eff_matrix, test_eff_matrix,
                                  err_msg="Incorrect efficiency matrix")
    pass
