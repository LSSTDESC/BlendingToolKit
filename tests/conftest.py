"""Configuration file for tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir():
    """Return the path to the data directory."""
    return Path(__file__).parent.joinpath("../data").resolve()
