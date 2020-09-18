import os
import galsim
import astropy.table


def load_catalog(catalog_name, selection_function=None, verbose=None, cosmos=False):
    """Returns astropy table with catalog name from input class.

    Args:
        catalog_name: File path of CatSim-like catalog or galsim COSMOS catalog to draw
                      galaxies from.
        selection_function: Selection cuts (if input) to place on input catalog.
        verbose: Whether to print information related to loading catalog.

    Returns:
        `astropy.table`: CatSim-like catalog with a selection criteria applied
        if provided.
    """

    if cosmos:
        cat = galsim.COSMOSCatalog(file_name=catalog_name)

    else:
        _, ext = os.path.splitext(catalog_name)
        fmt = "fits" if ext == ".fits" else "ascii.basic"
        cat = astropy.table.Table.read(catalog_name, format=fmt)
        table = astropy.table.Table.read(catalog_name, format="ascii.basic")

    # convert ra dec from degrees to arcsec in catalog.
    table["ra"] *= 3600
    table["dec"] *= 3600

    if verbose:
        print("Catalog loaded")
    if selection_function:
        if verbose:
            print("Selection criterion applied to input catalog")
        return selection_function(table)
    return table
