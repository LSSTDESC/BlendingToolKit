"""Implements the command-line interface for BTK."""
import hydra
from omegaconf import OmegaConf

from btk.main import main


@hydra.main(config_path="../conf", config_name="config")
def run(cfg: OmegaConf):
    """Implements CLI in BTK."""
    main(cfg)
