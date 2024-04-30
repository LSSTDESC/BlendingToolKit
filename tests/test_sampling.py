import btk

SEED = 0
import numpy as np


def test_density_sampling(data_dir):
    """Test new density sampling function."""
    catalog = btk.catalog.CatsimCatalog.from_file(data_dir / "input_catalog.fits")

    sampling_function = btk.sampling_functions.DensitySampling(
        max_number=50,
        min_number=1,
        stamp_size=24,
        max_shift=10,
        min_mag=-np.inf,
        max_mag=27.3,
        seed=SEED,
    )

    for _ in range(10):
        blends = sampling_function(catalog.get_raw_catalog())

        assert 1 <= len(blends) <= 50
        assert np.all(blends["ra"] >= -10) & np.all(blends["ra"] <= 10)
        assert np.all(blends["dec"] >= -10) & np.all(blends["dec"] <= 10)
