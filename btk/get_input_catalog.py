"""Function creates an astropy table containing information that is useful to
generate postage stamp images with appropriate distributions of shapes, colors,
fluxes, etc.
TODO:
1) Add script to load DC2 catalog
2) Add option to load multiple catlogs(eg star , galaxy)
"""
import os
import astropy.table


def load_catlog(catalog_name):
    """Returns astropy table with input name"""
    name, ext = os.path.splitext(catalog_name)
    if ext == '.fits':
        table = astropy.table.Table.read(catalog_name,
                                         format='fits')
    else:
        table = astropy.table.Table.read(catalog_name,
                                         format='ascii.basic')
    return table
