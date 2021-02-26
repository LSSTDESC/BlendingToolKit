# Contributing

Please makes sure to complete the following steps if you are interested in contributing to `BTK`: 

1. Start by git cloning the `BTK` repository: 

```
git clone https://github.com/LSSTDESC/BlendingToolKit.git
cd BlendingToolKit
```

2. You need to install the packages in both `dev-requirements.txt` (developer requirements) and `requirement.txt` (general requirements). It is recommend to have a separate environment (either using `virtualenv` or `conda`) when you are developing `BTK`. To install the requirements you can run:

```
pip install -r dev-requirements.txt
pip install -r requirements.txt
```

3. If you encounter problems during installation with `galsim` you might not have the correct pre-requisites. Please visit this [page](https://github.com/GalSim-developers/GalSim/blob/releases/2.2/INSTALL.rst) for instructions on how to install `galsim`. After successfully installing `galsim` try running step 2. again.

4. If you are planning to create a PR and merge it into BTK, it is recommended that you create a branch inside the `BTK` repo so that other people can contribute too :)

4. Every pull request must pass the workflows specified in `.github/workflows` before merging. 

    - The tool known as `pre-commit` will make it easy to for you to pass the linting workflow, install it in your local repository by running `pre-commit install` (it was installed from `dev-requirements.txt`)

    - For `BTK` we are using the `black` formatter, you can format your code by running `black .` which formats all python files accessible from your current directory. If you have an IDE that you like there are also [options](https://black.readthedocs.io/en/stable/editor_integration.html) to format on save.

    - You can run all the tests locally by simply running `pytest` or `pytest tests` inside your local repository.

5. If other branches were merged while you were working on this PR to the `main` branch, then you will to rebase before merging: 

```
git rebase origin/master
# follow the instructions and resolve conflicts...
git push --force
```

7. Finally ask for at least one approving review from @ismael-mendoza or other collaborators.
