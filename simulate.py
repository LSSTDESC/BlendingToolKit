#!/usr/bin/env python
"""Fast image simulation of blended objects. Script reads raw catalog, creates
a modified catalog of multiple (blended objects) and genrates PSF convolved
images. Objects and observing conditions are created with the
WeakLeensingDeblending package and drawn withGalsim.
"""
import argparse
import descwl
import btk


def main():
    # Initialize and parse command-line arguments.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--catalog_name', type=str, help='Name of catalog file \
        which must exist and be readable.')
    parser.add_argument('--max_number', type=int, default=2,
                        help='Maximum number of objects per blend [Default=2]')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of blends to be simulated per batch \
                        [Default=8]')
    parser.add_argument('--stamp_size', type=int, default=24,
                        help='Size of postage stamps in arcseconds [Default=24]')
    model_group = parser.add_argument_group('Source model options',
                                            'Specify options for\
                                            building source models \
                                            from catalog parameters.')
    descwl.model.GalaxyBuilder.add_args(model_group)
    survey_group = parser.add_argument_group('Survey parameters',
                                             'Specify survey camera and \
                                             observing parameters.')
    descwl.survey.Survey.add_args(survey_group)
    args = parser.parse_args()
    catalog = btk.get_input_catalog.load_catalog(args.catalog_name)
    blend_genrator = btk.create_blend_generator.generate(args, catalog)
    observing_genrator = btk.create_observing_generator.generate(args)


if __name__ == '__main__':
    main()