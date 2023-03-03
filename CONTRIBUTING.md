# Contributing

Please makes sure to complete the following steps if you are interested in contributing to `BTK`.

## Installing

1. Start by git cloning the `BTK` repository:

    ```bash
    git clone https://github.com/LSSTDESC/BlendingToolKit.git
    ```

2. It is recommended that you use `pdm` to manage the python environment for developing BTK. You can install pdm following the instructions [here](https://pdm.fming.dev/latest/#installation).

3. The following commands should install and activate a `btk` python environment managed by `pdm`:

```bash
cd BlendingToolKit

# install all python dependencies from pyproject.toml file
pdm install

# activate pdm environment
eval $(pdm venv activate in-project)

# install the git hook scripts
pre-commit install
```

If you have any problems with the installation, they are probably due to `galsim`. It is recommended that you follow the instructions for [installing galsim](https://galsim-developers.github.io/GalSim/_build/html/install.html) first (inside the `pdm` virtual environment), and then try again.

## Pull Requests

1. Every contribution to BTK must be made in a form of a Pull Request (PR) can eventually be merged to the `dev` branch. (NOTE: Only the maintainer(s) can push to the `main` branch)

2. Every pull request must pass the workflows specified in `.github/workflows` before merging.

    - The tool known as `pre-commit` will make it easy to for you to pass the linting workflow, install it in your local repository by running `pre-commit install`.

    - For `BTK` we are using the `black` formatter, you can format your code by running `black .` which formats all python files accessible from your current directory. If you have an IDE that you like there are also [options](https://black.readthedocs.io/en/stable/editor_integration.html) to format on save.

    - You can run all the tests locally by simply running `pytest` inside your local BTK repository.

    - If you get stuck by `pre-commit` errors, please ping us on your PR.

3. If other branches were merged while you were working on this PR to the `dev` branch, then you will to rebase before merging:

    ```bash
    git rebase origin/dev
    # follow the instructions and resolve conflicts...
    # Feel free to ask other developers if you are not sure of the conflicts.
    git push --force
    ```

4. Finally, ask for at least one approving review from [@ismael-mendoza](https://github.com/ismael-mendoza) or other collaborators.
