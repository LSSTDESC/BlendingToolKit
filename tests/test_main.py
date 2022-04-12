import subprocess

import pytest
from hydra import compose
from hydra import initialize

from btk.main import main


def get_cfg(overrides):
    overrides = [f"{key}={value}" for key, value in overrides.items()]
    with initialize(config_path="../conf"):
        cfg = compose("config", overrides=overrides)
    return cfg


def test_main():
    cfg = get_cfg(overrides={})
    main(cfg)

    # test survey CLI
    cfg = get_cfg(overrides={"surveys": ["LSST"]})
    main(cfg)
    cfg = get_cfg(overrides={"surveys": ["LSST", "DES"], "meas_band_name": ["r", "r"]})
    main(cfg)


def test_CLI():
    subprocess.run("btk", shell=True)
    subprocess.run("btk --help", shell=True)


def test_errors():

    cfg = get_cfg(overrides={"measure.measure_functions": ["NotExistantMeasureFunction"]})
    with pytest.raises(ValueError) as excinfo:
        main(cfg)
        assert "not implemented" in str(excinfo.value)
