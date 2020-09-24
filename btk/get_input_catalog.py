import os
import astropy.table


def load_catalog(catalog_name, selection_function=None, verbose=None):
    """Returns astropy table with catalog name from input class.

    Args:
        catalog_name: Name of CatSim-like catalog to draw galaxies from.
        selection_function: Selection cuts (if input) to place on input catalog.
        verbose: Whether to print information related to loading catalog.

    Returns:
        `astropy.table`: CatSim-like catalog with a selection criteria applied
        if provided.
    """
    name, ext = os.path.splitext(catalog_name)
    if ext == ".fits":
        table = astropy.table.Table.read(catalog_name, format="fits")
    else:
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
