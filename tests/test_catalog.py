from conftest import data_dir

from btk.catalog import CatsimCatalog, CosmosCatalog

CATALOG_PATH = str(data_dir / "sample_input_catalog.fits")
COSMOS_CATALOG_PATHS = [
    str(data_dir / "cosmos/real_galaxy_catalog_23.5_example.fits"),
    str(data_dir / "cosmos/real_galaxy_catalog_23.5_example_fits.fits"),
]


def test_reading_catsim_catalog():
    """Returns the catsim catalog"""

    catsim_catalog = CatsimCatalog.from_file(CATALOG_PATH)
    return catsim_catalog


def test_reading_cosmos_catalog():
    """Returns the cosmos catalog"""

    cosmos_catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
    return cosmos_catalog


def test_reading_cosmos_catalog_without_cuts():
    """Returns the cosmos catalog"""

    cosmos_catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS, exclusion_level="none")
    return cosmos_catalog


def test_getting_galsim_catalog():
    """Returns the galsim catalog"""

    cosmos_catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
    galsim_catalog = cosmos_catalog.get_galsim_catalog()
    # assert that the merged and galsim catalog have same length
    assert len(galsim_catalog) == len(cosmos_catalog.table), "error reading catalog with cuts"
    return galsim_catalog


def test_getting_galsim_catalog_without_cuts():
    """Returns the galsim catalog"""

    cosmos_catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS, exclusion_level="none")
    galsim_catalog = cosmos_catalog.get_galsim_catalog()
    return galsim_catalog


def test_verbose():
    """Testing the verbose option"""
    CatsimCatalog.from_file(CATALOG_PATH, verbose=True)
    CosmosCatalog.from_file(COSMOS_CATALOG_PATHS, verbose=True, exclusion_level="none")
