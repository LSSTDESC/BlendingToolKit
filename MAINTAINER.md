1. We use [poetry](https://python-poetry.org) as python package manager for BTK. It guarantees all developers are sharing the same python environment, makes it really easy to update dependencies, and publish to [pypi](https://pypi.org). Given some of the complications with installing `galsim` via `poetry`, we follow a hybrid approach of `conda`+`poetry`.

2. It is recommended to create a `conda` virtual environment (using `python3.7`) from scratch and use it to install all required dependencies. After having installed `conda`, please follow the following series of steps:

```
# enter to the local repo
cd BlendingToolKit

# create virtual environment.
conda create -n btk python=3.7
conda activate btk

# make sure you can import galsim after this step before moving on.
conda install fftw
conda install -c conda-forge eigen
conda install -c conda-forge galsim

# poetry allows us to install all dependencies directly from pyproject.toml (except galsim!)
# poetry reuses the current "btk" virtual environment from conda.
conda install -c conda-forge poetry
poetry install

# finally activate pre-commit
pre-commit install
```

In Ubuntu/Linux, you might getaway with simply running (and avoid having to use conda):

```
# inside your favorite python virtual environment...
sudo apt-get install libfftw3-dev libeigen3-dev
pip install --upgrade pip
pip install poetry
poetry install
pre-commit install
```

But I find the first method is more robust (works on a MAC too).

3. If any of the dependencies requires an update, you can simply run `poetry update` inside your local repo to automatically update and install them. Feel free to push the changes of the `pyproject.toml` or `poetry.lock` file to the PR you are working on.

4. You might also want to update the `requirements.txt` file every now and then:

```
poetry export -o requirements.txt --without-hashes --dev --extras "galsim-hub"
```

ideally everytime you run `poetry update`.
