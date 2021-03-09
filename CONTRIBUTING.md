# Contributing


## Installation

Please makes sure to complete the following steps if you are interested in contributing to `BTK`: 

1. Start by git cloning the `BTK` repository: 

```
git clone https://github.com/LSSTDESC/BlendingToolKit.git
cd BlendingToolKit
```

2. It is recommended to create a `conda` virtual environment (using `python3.8`) from scratch and use it to install all required dependencies. I find that the following steps work well for me (after having installed `conda`).

```

# create virtual environment.
conda create -n btk python=3.8
conda activate btk

# make sure you can import galsim after this step before moving on.
conda install fftw 
conda install -c conda-forge eigen
conda install -c conda-forge galsim 

# poetry allows us to install all dependencies directly from pyproject.toml (except galsim!)
# need to be inside your local BTK repo for this to work.
conda install -c conda-forge poetry
poetry install
```

In Ubuntu/Linux, you might getaway with simply running:

```
sudo apt-get install libfftw3-dev libeigen3-dev
pip install --upgrade pip
pip install poetry
poetry install
```

in your desired virtual environment. But I find the first method is more robust (works on a MAC too).

3. If you encounter problems during installation with `galsim` you might not have the correct pre-requisites. Please visit this [page](https://github.com/GalSim-developers/GalSim/blob/releases/2.2/INSTALL.rst) for instructions on how to install `galsim`. After successfully installing `galsim` try running step 2. again.

## Pull Requests

1. Every contribution to BTK must be made in a form of a Pull Request (PR) that can eventually be merged to the `main` branch. If you are planning to create a PR and merge it into BTK, it is recommended that you create a branch inside the `BTK` repo so that other people can contribute too :)

5. Every pull request must pass the workflows specified in `.github/workflows` before merging. 

    - The tool known as `pre-commit` will make it easy to for you to pass the linting workflow, install it in your local repository by running `pre-commit install`.

    - For `BTK` we are using the `black` formatter, you can format your code by running `black .` which formats all python files accessible from your current directory. If you have an IDE that you like there are also [options](https://black.readthedocs.io/en/stable/editor_integration.html) to format on save.

    - You can run all the tests locally by simply running `poetry run pytest` inside your local repository.

6. If other branches were merged while you were working on this PR to the `main` branch, then you will to rebase before merging: 

```
git rebase origin/main
# follow the instructions and resolve conflicts...
git push --force
```

7. Finally, ask for at least one approving review from [@ismael-mendoza](https://github.com/ismael-mendoza) or other collaborators.
