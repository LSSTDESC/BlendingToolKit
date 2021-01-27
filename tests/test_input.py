import os
import subprocess
import sys

from astropy.table import Table
import dill
import numpy as np
import pytest


@pytest.mark.skip
def test_input_draw(input_args, match_images):
    """Tests that objects are drawn correctly when btk is run with input config
    yaml file."""
    args = input_args()
    sys.path.append(os.getcwd())
    btk_input = __import__("btk_input")
    config_dict = btk_input.read_configfile(
        args.configfile, args.simulation, args.verbose
    )
    simulation_config_dict = config_dict["simulation"][args.simulation]
    user_config_dict = config_dict["user_input"]
    # Set seed
    np.random.seed(int(simulation_config_dict["seed"]))
    shifts = [
        [[-0.3, 1.2], [-1.6, -1.7]],
        [[-1.1, -2.1], [1.4, 1.8]],
        [[-1.8, -0.8], [-0.6, 2.2]],
        [[-2.0, -0.7], [-2.2, 1.9]],
        [[1.1, -1.5], [0.1, -2.3]],
        [[-2.3, 1.9], [0.4, -1.9]],
        [[2.0, -2.0], [2.0, 0.1]],
        [[0.2, 2.4], [-1.8, -2.0]],
    ]
    indexes = [[4, 5], [9, 1], [9, 2], [0, 2], [3, 8], [0, 7], [10, 2], [0, 10]]
    draw_blend_generator = btk_input.make_draw_generator(
        user_config_dict, simulation_config_dict, shifts=shifts, indexes=indexes
    )
    draw_output = next(draw_blend_generator)
    assert len(draw_output["blend_list"]) == 8, "Default batch should return 8"
    assert (
        len(draw_output["blend_list"][3]) < 3
    ), "Default max_number should \
        generate 2 or 1 galaxies per blend."
    assert (
        draw_output["obs_condition"][0].survey_name == "LSST"
    ), "Default \
        observing survey is LSST."
    match_images.match_blend_images_default(draw_output["blend_images"])
    match_images.match_isolated_images_default(draw_output["isolated_images"])
    pass


def check_output_file(user_config_dict, simulation):
    """Check if metrics output is correctly written to file.

    Args:
        simulation: is a string indicating name of simulated blend scene
                    like "two_gal" or "group".
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
    """
    output_path = os.path.join(
        user_config_dict["output_dir"], user_config_dict["output_name"]
    )
    if not os.path.isdir(output_path):
        raise FileNotFoundError(f"btk output must be saved at {output_path}")
    output_name = os.path.join(output_path, simulation + "_metrics_results.dill")
    if not os.path.isfile(output_name):
        raise FileNotFoundError(f"btk output must be saved at {output_name}")
    output_configfile = os.path.join(output_path, simulation + "_config.yaml")
    if not os.path.isfile(output_configfile):
        raise FileNotFoundError(
            f"config values to run btk must be saved at \
                                {output_name}"
        )
    pass


def check_output_values(user_config_dict, simulation):
    """Check if metrics output results written to file are in the correct
    format.

    Args:
        simulation: is a string indicating name of simulated blend scene
                    like "two_gal" or "group".
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
    """
    output_path = os.path.join(
        user_config_dict["output_dir"], user_config_dict["output_name"]
    )
    output_name = os.path.join(output_path, simulation + "_metrics_results.dill")
    with open(output_name, "rb") as handle:
        results = dill.load(handle)
    result_keys = ["detection", "segmentation", "flux", "shapes"]
    assert set(results.keys()) == set(result_keys), (
        "Results have incorrect"
        f"keys. Found "
        f"{results.keys()}, "
        f"expected {result_keys}"
    )
    if not isinstance(results["detection"][0], Table):
        raise ValueError(
            "Expected astropy table in results['detection'][0],  "
            f"got {type(results['detection'][0])} "
        )
    if not isinstance(results["detection"][1], Table):
        raise ValueError(
            "Expected astropy table in results['detection'][1],  "
            f"got {type(results['detection'][1])} "
        )
    if not isinstance(results["detection"][2], list):
        raise ValueError(
            "Expected astropy table in results['detection'][2],  "
            f"got {type(results['detection'][2])} "
        )
    pass


def delete_output_file(user_config_dict, simulation):
    """Deletes metric output results written to file.

    Args:
        simulation: is a string indicating name of simulated blend scene
                    like "two_gal" or "group".
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
    """
    output_path = os.path.join(
        user_config_dict["output_dir"], user_config_dict["output_name"]
    )
    output_name = os.path.join(output_path, simulation + "_metrics_results.dill")
    yaml_output_name = os.path.join(output_path, simulation + "_config.yaml")
    if os.path.isfile(yaml_output_name):
        subprocess.call(["rm", yaml_output_name])
    if os.path.isfile(output_name):
        subprocess.call(["rm", output_name])
    if os.path.isdir(output_path):
        subprocess.call(["rmdir", output_path])
    return


@pytest.mark.skip
def test_input_output(input_args):
    """Checks output of btk called in test_input for input simulation.

    Checks that the output files have correct values and are saved in correct
    format and location.

    """
    for simulation in ["two_gal", "multi_gal", "group"]:
        command = ["python3", "btk_input.py", "--configfile", "tests/test-config.yaml"]
        subprocess.call(command + ["--simulation", simulation])
        args = input_args(simulation=simulation)
        sys.path.append(os.getcwd())
        btk_input = __import__("btk_input")
        config_dict = btk_input.read_configfile(
            args.configfile, args.simulation, args.verbose
        )
        user_config_dict = config_dict["user_input"]
        try:
            # Check if results were written to file.
            check_output_file(user_config_dict, simulation)
            # Check output was written to file in the right format.
            check_output_values(user_config_dict, simulation)
            print("btk output files created")
        except FileNotFoundError as e:
            print("btk output files not found")
            raise e
        except ValueError as e:
            print("btk output files were not in correct format")
            raise e
        finally:
            # Delete unit test output files.
            delete_output_file(user_config_dict, simulation)
            print("deleted files")
        pass


def basic_meas(
    user_config_dict,
    simulation_config_dict,
    btk_input,
):
    """Checks if detection output from the default meas generator  matches
    the pre-computed value .

    The outputs from basic meas generator were visually checked and verified.
    This function makes sure that and changes made to the btk pipeline will not
    affect the detection results.

    Args:
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        simulation_config_dict (dict): Dictionary which sets the parameter
        btk_input : Module that runs btk for an input config file.
    """
    np.random.seed(int(simulation_config_dict["seed"]))
    test_detect_centers = [
        [[66, 69]],
        [[48, 67]],
        [[56, 54]],
        [[53, 57]],
    ]
    shifts = [
        [[-2.4, -0.8, 0.9, 1.4], [-2.3, -0.4, 2.3, 1.9]],
        [[-2.3, 2.0, 0.0, 0.4, 0.7], [1.6, 0.1, 0.7, 0.9, 2.3]],
        [[0.6, -0.6, 1.7, 0.4, 2.3, 0.2], [-1.7, -1.1, -1.6, 0.7, 1.0, -1.5]],
        [[-1.3, -1.0, 1.2, -2.3], [-0.2, -0.9, -1.8, 1.4]],
    ]
    indexes = [
        [
            3,
            1,
            9,
            6,
        ],
        [6, 10, 3, 7, 4],
        [10, 0, 7, 1, 9, 4],
        [1, 3, 2, 8],
    ]

    draw_blend_generator = btk_input.make_draw_generator(
        user_config_dict, simulation_config_dict, shifts=shifts, indexes=indexes
    )
    measure_generator = btk_input.make_measure_generator(
        user_config_dict, draw_blend_generator
    )

    output, deb, _ = next(measure_generator)
    for i in range(len(output["blend_list"])):
        detected_centers = deb[i]["peaks"]

        np.testing.assert_array_almost_equal(
            detected_centers,
            test_detect_centers[i],
            decimal=3,
            err_msg="Did not get desired detected_centers",
        )
    pass


def sep_meas(
    user_config_dict,
    simulation_config_dict,
    btk_input,
):
    """Checks if detection output from the sep meas generator  matches
    the pre-computed value .

    The outputs from basic meas generator were visually checked and verified.
    This function makes sure that and changes made to the btk pipeline will not
    affect the detection results.

    Args:
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        simulation_config_dict (dict): Dictionary which sets the parameter
        btk_input : Module that runs btk for an input config file.
    """
    np.random.seed(int(simulation_config_dict["seed"]))
    test_detect_centers = [
        [[55.343, 57.336], [66.363, 69.078]],
        [[69.635, 59.765], [60.169, 63.402], [48.093, 67.462]],
        [[57.185, 54.397]],
        [[65.329, 50.522], [53.786, 57.411]],
    ]
    shifts = [
        [[-2.4, -0.8, 0.9, 1.4], [-2.3, -0.4, 2.3, 1.9]],
        [[-2.3, 2.0, 0.0, 0.4, 0.7], [1.6, 0.1, 0.7, 0.9, 2.3]],
        [[0.6, -0.6, 1.7, 0.4, 2.3, 0.2], [-1.7, -1.1, -1.6, 0.7, 1.0, -1.5]],
        [[-1.3, -1.0, 1.2, -2.3], [-0.2, -0.9, -1.8, 1.4]],
    ]
    indexes = [
        [
            3,
            1,
            9,
            6,
        ],
        [6, 10, 3, 7, 4],
        [10, 0, 7, 1, 9, 4],
        [1, 3, 2, 8],
    ]

    draw_blend_generator = btk_input.make_draw_generator(
        user_config_dict, simulation_config_dict, shifts=shifts, indexes=indexes
    )
    user_config_dict["utils_input"]["measure_function"] = "SEP_params"
    measure_generator = btk_input.make_measure_generator(
        user_config_dict, draw_blend_generator
    )

    output, deb, _ = next(measure_generator)
    for i in range(len(output["blend_list"])):
        detected_centers = deb[i]["peaks"]
        np.testing.assert_array_almost_equal(
            detected_centers,
            test_detect_centers[i],
            decimal=3,
            err_msg="Did not get desired detected_centers",
        )
    pass


def stack_meas(
    user_config_dict,
    simulation_config_dict,
    btk_input,
):
    """Checks if detection output from the stack meas generator  matches
    the pre-computed value .

    The outputs from basic meas generator were visually checked and verified.
    This function makes sure that and changes made to the btk pipeline will not
    affect the detection results.

    Args:
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        simulation_config_dict (dict): Dictionary which sets the parameter
        btk_input : Module that runs btk for an input config file.
    """
    np.random.seed(int(simulation_config_dict["seed"]))
    test_detect_dx = [
        [56.16308227, 62.96011953, 55.99366715, 48.97120018],
        [48.95804179],
        [58.98873963],
        [60.03759286, 53.9629357, 68.12901647],
    ]
    test_detect_dy = [
        [67.09899307, 58.19916427, 59.86796513, 55.89051818],
        [67.0042175],
        [66.97054341],
        [47.07838961, 62.05724329, 61.94601734],
    ]
    shifts = [
        [[-2.4, -0.8, 0.9, 1.4], [-2.3, -0.4, 2.3, 1.9]],
        [[-2.3, 2.0, 0.0, 0.4, 0.7], [1.6, 0.1, 0.7, 0.9, 2.3]],
        [[0.6, -0.6, 1.7, 0.4, 2.3, 0.2], [-1.7, -1.1, -1.6, 0.7, 1.0, -1.5]],
        [[-1.3, -1.0, 1.2, -2.3], [-0.2, -0.9, -1.8, 1.4]],
    ]
    indexes = [
        [
            3,
            1,
            9,
            6,
        ],
        [6, 10, 3, 7, 4],
        [10, 0, 7, 1, 9, 4],
        [1, 3, 2, 8],
    ]

    draw_blend_generator = btk_input.make_draw_generator(
        user_config_dict, simulation_config_dict, shifts=shifts, indexes=indexes
    )
    user_config_dict["utils_input"]["measure_function"] = "Stack_params"
    measure_generator = btk_input.make_measure_generator(
        user_config_dict, draw_blend_generator
    )

    output, deb, meas = next(measure_generator)
    for i in range(len(output["blend_list"])):
        detected_center_x = meas[i]["base_NaiveCentroid_x"]
        detected_center_y = meas[i]["base_NaiveCentroid_y"]
        np.testing.assert_array_almost_equal(
            detected_center_x,
            test_detect_dx[i],
            decimal=3,
            err_msg="Did not get desired detected_centers",
        )
        np.testing.assert_array_almost_equal(
            detected_center_y,
            test_detect_dy[i],
            decimal=3,
            err_msg="Did not get desired detected_centers",
        )
    pass


def scarlet_meas(user_config_dict, simulation_config_dict, btk_input):
    """Checks if detection output from the scarlet meas generator matches
    the pre-computed value .

    The outputs from basic meas generator were visually checked and verified.
    This function makes sure that and changes made to the btk pipeline will not
    affect the detection results.

    Args:
        user_config_dict: Dictionary with information to run user defined
            functions (filenames, file location of user algorithms).
        simulation_config_dict (dict): Dictionary which sets the parameter
        btk_input : Module that runs btk for an input config file.
    """

    np.random.seed(int(simulation_config_dict["seed"]))
    test_detect_centers = [
        [[58.063703, 59.749699], [61.157868, 69.30290], [68.304245, 61.537312]],
        [[59.915507, 50.167592], [65.766700, 65.105297]],
        [[51.243380, 58.382503], [54.900160, 68.5794316]],
        [[70.645195, 51.627339], [63.226545, 56.1251558]],
    ]
    shifts = [
        [[-2.4, -0.8, 0.9, 1.4], [-2.3, -0.4, 2.3, 1.9]],
        [[-2.3, 2.0, 0.0, 0.4, 0.7], [1.6, 0.1, 0.7, 0.9, 2.3]],
        [[0.6, -0.6, 1.7, 0.4, 2.3, 0.2], [-1.7, -1.1, -1.6, 0.7, 1.0, -1.5]],
        [[-1.3, -1.0, 1.2, -2.3], [-0.2, -0.9, -1.8, 1.4]],
    ]
    indexes = [
        [
            3,
            1,
            9,
            6,
        ],
        [6, 10, 3, 7, 4],
        [10, 0, 7, 1, 9, 4],
        [1, 3, 2, 8],
    ]

    draw_blend_generator = btk_input.make_draw_generator(
        user_config_dict, simulation_config_dict, shifts=shifts, indexes=indexes
    )
    user_config_dict["utils_input"]["measure_function"] = "Scarlet_params"
    measure_generator = btk_input.make_measure_generator(
        user_config_dict, draw_blend_generator
    )

    output, deb, _ = next(measure_generator)
    for i in range(len(output["blend_list"])):
        detected_centers = deb[i]["peaks"]
        np.testing.assert_array_almost_equal(
            detected_centers,
            test_detect_centers[i],
            decimal=3,
            err_msg="Did not get desired detected_centers",
        )
    pass


@pytest.mark.skip
def test_measure(input_args):
    """Performs measurements for different measurement functions and
    simulations, and checks that the output matches previously measured values.
    """
    args = input_args()
    sys.path.append(os.getcwd())
    btk_input = __import__("btk_input")
    config_dict = btk_input.read_configfile(
        args.configfile, args.simulation, args.verbose
    )
    simulation_config_dict = config_dict["simulation"][args.simulation]
    simulation_config_dict["max_number"] = 6
    simulation_config_dict["batch_size"] = 4

    user_config_dict = config_dict["user_input"]
    basic_meas(
        user_config_dict,
        simulation_config_dict,
        btk_input,
    )
    try:
        sep_meas(user_config_dict, simulation_config_dict, btk_input)
    except ImportError:
        print("sep not found")
    try:
        stack_meas(user_config_dict, simulation_config_dict, btk_input)
    except ImportError:
        print("stack not found")
    pass


def basic_metric_two_gal(output_name):
    """Loads metric results dill file and compares it to target value"""
    with open(output_name, "rb") as handle:
        results = dill.load(handle)
    detected_metrics = np.array(results["detection"][2])
    test_metric_summary = np.array(
        [
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [2, 1, 1, 0, 0, 1, 1, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [2, 1, 1, 0, 0, 1, 1, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
        ]
    )
    np.testing.assert_array_almost_equal(
        detected_metrics,
        test_metric_summary,
        decimal=3,
        err_msg="Did not get desired detection metrics summary",
    )
    pass


def basic_metric_two_gal_multi(output_name):
    """Loads metric results dill file and compares it to target value"""
    with open(output_name, "rb") as handle:
        results = dill.load(handle)
    detected_metrics = np.array(results["detection"][2])
    test_metric_summary = np.array(
        [
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0],
            [2, 1, 1, 0, 0, 1, 1, 0, 0],
            [2, 1, 1, 0, 0, 1, 1, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [2, 2, 0, 0, 0, 2, 0, 0, 0],
            [2, 1, 1, 0, 0, 1, 1, 0, 0],
        ]
    )
    np.testing.assert_array_almost_equal(
        detected_metrics,
        test_metric_summary,
        decimal=3,
        err_msg="Did not get desired detection metrics summary",
    )
    pass
