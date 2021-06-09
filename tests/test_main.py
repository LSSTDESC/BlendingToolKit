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
