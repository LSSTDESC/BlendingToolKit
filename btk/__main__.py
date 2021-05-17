"""Implements command-line interface for BTK."""

from omegaconf import OmegaConf


def get_sampling_function(cfg: OmegaConf):
    if cfg.sampling_function == "default":
        pass


def main():
    pass
