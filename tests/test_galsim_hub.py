from conftest import data_dir

from btk.catalog import CosmosCatalog
from btk.draw_blends import GalsimHubGenerator
from btk.sampling_functions import DefaultSamplingGalsimHub
from btk.survey import get_surveys

COSMOS_CATALOG_PATHS = [
    data_dir / "cosmos/real_galaxy_catalog_23.5_example.fits",
    data_dir / "cosmos/real_galaxy_catalog_23.5_example_fits.fits",
]


def draw_galsim_hub():
    stamp_size = 24.0
    batch_size = 2
    catalog = CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
    sampling_function = DefaultSamplingGalsimHub(stamp_size=stamp_size)

    draw_generator = GalsimHubGenerator(
        catalog,
        sampling_function,
        get_surveys("HST"),
        batch_size=batch_size,
        stamp_size=stamp_size,
        cpus=1,
        add_noise=True,
        verbose=True,
        meas_bands=["f814w"],
    )

    _ = next(draw_generator)
