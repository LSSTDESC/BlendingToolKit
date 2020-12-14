import pytest

from btk.catalog import WLDCatalog

CATALOG_PATH = "data/sample_input_catalog.fits"


def test_verbose():
    """For coverage"""
    WLDCatalog.from_file(CATALOG_PATH, verbose=True)


def test_apply_selection_function():
    catalog = WLDCatalog.from_file(CATALOG_PATH)
    callable_selection_function = lambda table: table
    catalog.apply_selection_function(callable_selection_function)

    non_callable_selection_function = 5
    with pytest.raises(TypeError):
        catalog.apply_selection_function(non_callable_selection_function)
