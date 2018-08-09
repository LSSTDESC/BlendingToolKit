import descwl
import argparse


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    survey_group = parser.add_argument_group('Survey parameters',
        'Specify survey camera and observing parameters.')
    descwl.survey.Survey.add_args(survey_group)
    args = parser.parse_args()
    keys = ['exposure_time', 'cosmic_shear_g2', 'image_width', 'filter_band',
            'image_height', 'zenith_psf_fwhm', 'survey_name']
    kys2 = ['exposure-time', 'cosmic-shear-g2', 'image-width', 'filter-band',
            'image-height', 'zenith-psf-fwhm', 'survey-name']
    survey = descwl.survey.Survey.from_args(args)
    for i, key in enumerate(keys):
        print(str(kys2[i]), str(vars(survey)[key]))


if __name__ == "__main__":
    main()
