import pytest
import subprocess
import imp

# check output dump file
# delete test output at end


class Input_Args(object):
    def __init__(self, simulation='two_gal', name='unit_test',
                 configfile='tests/test-config.yaml', verbose=True):
        self.simulation = simulation
        self.configfile = configfile
        self.name = name
        self.verbose = verbose


def test_input_draw():
    args = Input_Args()
    btk_input = __import__('btk_input')
    config_dict = btk_input.read_configfile(args.configfile, args.simulation,
                                            args.verbose)
    simulation_config_dict = config_dict['simulation'][args.simulation]
    user_config_dict = config_dict['user_input']
    draw_blend_generator = btk_input.make_draw_generator(
            args, user_config_dict, simulation_config_dict)
    draw_output = next(draw_blend_generator)
    assert len(draw_output['blend_list']) == 8, "Default batch should return 8"
    assert len(draw_output['blend_list'][3]) < 3, "Default max_number should \
        generate 2 or 1 galaxies per blend."
    assert draw_output['obs_condition'][5][0].survey_name == 'LSST', "Default \
        observing survey is LSST."
    test_draw = imp.load_source("", 'tests/test_draw.py')
    #test_draw = __import__('tests.test_draw')
    test_draw.match_blend_images_default(draw_output['blend_images'])
    test_draw.match_isolated_images_default(draw_output['isolated_images'])
    pass


@pytest.mark.timeout(35)
def test_input():
    import sys
    sys.path.append("..")
    """
    tests if btk_input script is correctly executed
    """
    command = ['python', 'btk_input.py', '--name', 'unit_test',
               '--configfile', 'tests/test-config.yaml']
    for s in ['all', 'two_gal', 'multi_gal', 'group']:
        subprocess.check_call(command + ['--simulation', s])
