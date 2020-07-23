import os
import astropy.table


def load_catalog(Args, selection_function=None):
    """Returns astropy table with catalog name from input class.

    Args:
        Args: Class containing input parameters.
        Args.catalog_name: Name of CatSim-like catalog to draw galaxies from.
        selection_function: Selection cuts (if input) to place on input catalog.

    Returns:
        `astropy.table`: CatSim-like catalog with a selection criteria applied
        if provided.
    """
    name, ext = os.path.splitext(Args.catalog_name)
    if ext == ".fits":
        table = astropy.table.Table.read(Args.catalog_name, format="fits")
    else:
        table = astropy.table.Table.read(Args.catalog_name, format="ascii.basic")
    if Args.verbose:
        print("Catalog loaded")
    if selection_function:
        if Args.verbose:
            print("Selection criterion applied to input catalog")
        return selection_function(table)
    return table
