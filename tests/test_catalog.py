from btk.catalog import CatsimCatalog

CATALOG_PATH = "data/sample_input_catalog.fits"


def test_verbose():
    # for coverage.
    CatsimCatalog.from_file(CATALOG_PATH, verbose=True)
