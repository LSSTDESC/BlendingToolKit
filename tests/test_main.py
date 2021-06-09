import subprocess

import pytest
from hydra.experimental import compose
from hydra.experimental import initialize

from btk.main import main


def get_cfg(overrides):
    overrides = [f"{key}={value}" for key, value in overrides.items()]
    with initialize(config_path="../conf"):
        cfg = compose("config", overrides=overrides)
    return cfg


def test_main():
    cfg = get_cfg(overrides={})
    main(cfg)


def test_CLI():
    subprocess.run("btk", shell=True)
    subprocess.run("btk --help", shell=True)


def test_errors():

    cfg = get_cfg(overrides={"catalog.name": "MyCatalog"})
    with pytest.raises(ValueError) as excinfo:
        main(cfg)
        assert "not implemented" in str(excinfo.value)

    cfg = get_cfg(overrides={"sampling.name": "MySampling"})
    with pytest.raises(ValueError) as excinfo:
        main(cfg)
        assert "not implemented" in str(excinfo.value)

    cfg = get_cfg(overrides={"surveys": ["survey1", "survey2"]})
    with pytest.raises(ValueError) as excinfo:
        main(cfg)
        assert "not implemented" in str(excinfo.value)

    cfg = get_cfg(overrides={"draw_blends.name": "MyDrawBlends"})
    with pytest.raises(ValueError) as excinfo:
        main(cfg)
        assert "not implemented" in str(excinfo.value)
