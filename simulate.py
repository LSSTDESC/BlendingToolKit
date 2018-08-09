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
    survey_group = parser.add_argument_group('Survey parameters',
                                             'Specify survey camera and \
                                             observing parameters.')
    descwl.survey.Survey.add_args(survey_group)
    parser.add_argument('--catalog_name', type=str, help='Name of catalog file \
        which must exist and be readable.')
    parser.add_argument('--max_number', type=str, help='Maximum number of \
        objects per blend')
    parser.add_argument('--batch_size', type=str, help='Number of blends\
        to be simulated per batch')
    model_group = parser.add_argument_group('Source model options',
                                            'Specify options for\
                                            building source models \
                                            from catalog parameters.')
    descwl.model.GalaxyBuilder.add_args(model_group)
    model_star_group = parser.add_argument_group('Source model options',
                                                 'Specify options for\
                                                 building source models \
                                                 from catalog parameters.')
    descwl.model.StarBuilder.add_args(model_star_group)
    args = parser.parse_args()
    table = btk.get_input_catalog.load_catalog(args)
    blend_genrator = btk.create_blend_generator.genrate(Args)
        if args.star_catalog_name!=None:
            star_catalog = descwl.catalog.ReaderStar.from_args(args)
        if args.verbose:
            if args.catalog_name!=None:
                print('Read %d catalog entries from %s' % (len(catalog.table),catalog.catalog_name))
            if args.star_catalog_name!=None:
                print('Read %d catalog entries from %s' % (len(star_catalog.table),star_catalog.star_catalog_name))
        survey = descwl.survey.Survey.from_args(args)
        if args.verbose:
            print(survey.description())
        if args.catalog_name!=None:
            galaxy_builder = descwl.model.GalaxyBuilder.from_args(survey,args)
        if args.star_catalog_name!=None:
            star_builder = descwl.model.StarBuilder.from_args(survey,args)

        render_engine = descwl.render.Engine.from_args(survey,args)
        if args.verbose:
            print(render_engine.description())

        analyzer = descwl.analysis.OverlapAnalyzer(survey,args.no_hsm, not args.add_lmfit, args.add_noise)

        output = descwl.output.Writer.from_args(survey,args)
        if args.verbose:
            print(output.description())

        trace('initialized')
        if args.catalog_name!=None:
            for entry,dx,dy in catalog.potentially_visible_entries(survey,render_engine):

                try:
                    galaxy = galaxy_builder.from_catalog(entry,dx,dy,survey.filter_band)
                    stamps,bounds = render_engine.render_galaxy(galaxy,args.no_partials,args.calculate_bias)
                    analyzer.add_galaxy(galaxy,stamps,bounds)
                    trace('render')

                except (descwl.model.SourceNotVisible,descwl.render.SourceNotVisible):
                    pass
        if args.star_catalog_name!=None:
            for entry,dx,dy in star_catalog.potentially_visible_entries(survey,render_engine):

                try:
                    star = star_builder.from_catalog(entry,dx,dy,survey.filter_band)
                    stamps,bounds = render_engine.render_star(star)
                    analyzer.add_star(star,stamps,bounds)
                    trace('render')

                except (descwl.model.SourceNotVisible,descwl.render.SourceNotVisible):
                    pass

        results = analyzer.finalize(args.verbose,trace,args.calculate_bias)
        output.finalize(results,trace)

    except RuntimeError as e:
        print(str(e))
        return -1

if __name__ == '__main__':
    main()