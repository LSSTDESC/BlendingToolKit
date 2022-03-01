# Contributing

Please makes sure to complete the following steps if you are interested in contributing to `BTK`.

## Installing

1. Start by git cloning the `BTK` repository:

    ```bash
    git clone https://github.com/LSSTDESC/BlendingToolKit.git
    ```

2. It is highly recommended that you use `conda+poetry` to manage the python environment for developing BTK. The following commands should install and activate a `btk` python environment managed by `poetry`:

```bash
cd BlendingToolKit

# you can skip this step if you already have a new conda environment specifically for BTK.
conda create -n btk37 python=3.7

# activate conda environment.
conda activate btk37

# install poetry
conda install -c conda-forge poetry

# install BTK dependencies from pyproject.toml file
poetry install

# activate poetry environment from inside the conda environment.
poetry shell

# install the git hook scripts
pre-commit install
```

If you have any problems with the installation, they are probably due to `galsim`. It is recommended that you follow the instructions for [installing galsim](https://galsim-developers.github.io/GalSim/_build/html/install.html) first (inside the `btk37` conda environment), and then try again.

## Pull Requests

1. Every contribution to BTK must be made in a form of a Pull Request (PR) that can eventually be merged to the `main` branch.

2. Every pull request must pass the workflows specified in `.github/workflows` before merging.

    - The tool known as `pre-commit` will make it easy to for you to pass the linting workflow, install it in your local repository by running `pre-commit install`.

    - For `BTK` we are using the `black` formatter, you can format your code by running `black .` which formats all python files accessible from your current directory. If you have an IDE that you like there are also [options](https://black.readthedocs.io/en/stable/editor_integration.html) to format on save.

    - You can run all the tests locally by simply running `pytest` inside your local repository.

3. If other branches were merged while you were working on this PR to the `main` branch, then you will to rebase before merging:

    ```bash
    git rebase origin/main
    # follow the instructions and resolve conflicts...
    # Feel free to ask other developers if you are not sure of the conflicts.
    git push --force
    ```

4. Finally, ask for at least one approving review from [@ismael-mendoza](https://github.com/ismael-mendoza) or other collaborators.
