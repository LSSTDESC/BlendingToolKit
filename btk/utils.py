"""Contains utility functions, including functions for loading saved results."""
from astropy.table import Table
from astropy.wcs import WCS

DEFAULT_SEED = 0


def add_pixel_columns(catalog: Table, wcs: WCS):
    """Uses the wcs to add `x_peak` and `y_peak` (pixel centroids) columns to the catalog.

    The catalog must contain `ra` and `dec` columns.

    Args:
        catalog (astropy.table.Table): Catalog to modify.
        wcs (astropy.wcs.WCS): WCS corresponding to the wanted
                               transformation.
    """
    for blend in catalog:
        x_peak = []
        y_peak = []
        for gal in blend:
            coords = wcs.world_to_pixel_values(gal["ra"] / 3600, gal["dec"] / 3600)
            x_peak.append(coords[0])
            y_peak.append(coords[1])
        blend.add_column(x_peak, name="x_peak")
        blend.add_column(y_peak, name="y_peak")
