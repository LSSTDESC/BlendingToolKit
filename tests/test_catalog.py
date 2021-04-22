from conftest import data_dir

from btk.catalog import CatsimCatalog
from btk.catalog import CosmosCatalog

CATALOG_PATH = (data_dir / "sample_input_catalog.fits").as_posix()
COSMOS_CATALOG_PATHS = [
    (data_dir / "cosmos/real_galaxy_catalog_23.5_example.fits").as_posix(),
    (data_dir / "cosmos/real_galaxy_catalog_23.5_example_fits.fits").as_posix(),
]


def test_reading_catsim_catalog():
    """Returns the catsim catalog"""

    catsim_catalog = CatsimCatalog.from_file(CATALOG_PATH)
    return catsim_catalog


def test_reading_cosmos_catalog():
    """Returns the cosmos catalog"""

    cosmos_catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
    return cosmos_catalog


def test_getting_galsim_catalog():
    """Returns the galsim catalog"""

    cosmos_catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
    galsim_catalog = cosmos_catalog.get_galsim_catalog()
    return galsim_catalog


def test_verbose():
    """Testing the verbose option"""
    CatsimCatalog.from_file(CATALOG_PATH, verbose=True)
    CosmosCatalog.from_file(COSMOS_CATALOG_PATHS, verbose=True)
