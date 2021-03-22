from btk.catalog import CatsimCatalog
from btk.catalog import CosmosCatalog

CATALOG_PATH = "data/sample_input_catalog.fits"
COSMOS_CATALOG_PATHS = ["data/real_galaxy_catalog_25.2.fits", "data/real_galaxy_catalog_25.2_fits.fits"]


def test_reading_catsim_catalog():
    """Returns the catsim catalog"""

    catsim_catalog = CatsimCatalog.from_file(CATALOG_PATH)
    return catsim_catalog
    

def test_reading_cosmos_catalog():
    """Returns the cosmos catalog"""

    cosmos_catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
    return cosmos_catalog

    
def test_verbose():
    # for coverage.
    CatsimCatalog.from_file(CATALOG_PATH, verbose=True)
