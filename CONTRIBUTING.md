# Contributing to BlendingToolKit

Everyone is welcome to contribute to this project, and contributions can take many different forms, from helping to answer questions on the [discussions page](https://github.com/LSSTDESC/BlendingToolKit/discussions), to contributing to the code-base by making a Pull Request.

In order to foster an open, inclusive, and welcoming community, all contributors agree to adhere to [BlendingToolKit code of conduct](CODE_OF_CONDUCT.md).

## Contributing code using Pull Requests

Code contributions are most welcome. You can in particular look for GitHub issues marked as `contributions welcome` or `good first issue`. But you can also propose adding a new functionality, in which case you may find it beneficial to first open a GitHub issue to discuss the feature you want to implement ahead of opening a Pull Request.

Once you have some code you wish to contribute, you can follow this procedure to prepare a Pull Request:

- Fork the `BTK` repository under your own account, using the **Fork** button on the top right of the [GitHub page](https://github.com/LSSTDESC/BlendingToolKit).

- We use [poetry](https://python-poetry.org/docs/) for managing the dependencies of the project. You can install it with:

  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```

- Clone and pip install your fork of the repository like so:

  ```bash
  git clone https://github.com/YOUR_USERNAME/BlendingToolKit
  cd BlendingToolKit
  poetry install
  poetry shell # activate the virtual environment
  ```

  This will install your local fork in editable mode, meaning you can directly modify source files in this folder without having to reinstall the package for them to be taken into account.

- Open a branch for your developments:

  ```bash
  git checkout -b name-that-describes-my-feature
  ```

  It's usually a good idea to name your branch after the issue you are working on, if any. For example:

    ```bash
    git checkout -b i/42
    ```

- Add your changes to the code using your favorite editor. You may at any moment test that everything is still working by running the test suite. From the root folder of the repository, run:

  ```bash
  poetry run pytest # or poetry shell; pytest
  ```

- Once you are happy with your modifications, commit them, and push your changes to GitHub:

  ```python
  git add file_I_changed.py
  git commit -m "a message that describes your modifications"
  git push -set-upstream origin name-that-describes-my-feature
  ```

- From your GitHub interface, you should now be able to open a Pull Request to the BlendingToolKit repository.

Before submitting your PR, have a look at the procedure documented below.

### Checklist before opening a Pull Request

- Pull Requests should be self-contained and limited in scope, otherwise they become too difficult to review. If your modifications are broad, consider opening several smaller Pull Requests.

- Make sure your fork and branch are up-to-date with the `dev` branch of BTK. To update your local branch, you may do so from the GitHub interface, or you may use this CLI command:

  ```bash
  git remote add upstream http://www.github.com/GalSim-developers/BlendingToolKit # Only needs to be done once.

  git fetch upstream
  git rebase upstream/dev # This will update your local branch with the latest changes from the dev branch.
  ```

- Make sure the unit tests still work:

  ```bash
  poetry run pytest
  ```

  Ideally there should be some new unit tests for the new functionality, unless the work is completely covered by existing unit tests.

- Make sure your code conforms to the [Black](https://github.com/psf/black) style:

  ```bash
  black .
  ```

- If your changes contain multiple commits, we encourage you to squash them into a single (or very few) commit, before opening the PR. To do so, you can using this command:

```bash
git rebase -i
```

### Opening the Pull Request

- On the GitHub site, go to "Code". Then click the green "Compare and Review" button. Your branch is probably in the "Example Comparisons" list, so click on it. If not, select it for the "compare" branch.

- Make sure you are comparing your new branch to the upstream `dev`. Press Create Pull Request button.

- Give a brief title. (We usually leave the branch number as the start of the title.)

- Explain the major changes you are asking to be code reviewed. Often it is useful to open a second tab in your browser where you can look through the diff yourself to remind yourself of all the changes you have made.

### After submitting the pull request

- Check to make sure that the PR can be merged cleanly. If it can, GitHub will report that "This branch has no conflicts with the base branch." If it doesn't, then you need to merge from master into your branch and resolve any conflicts.

- Wait a few minutes for the continuous integration tests to be run. Then make sure that the tests reports no errors. If not, click through to the details and try to figure out what is causing the error and fix it.

### After code review

- Once at least 1 and preferably 2 people have reviewed the code, and you have responded to all of their comments, we generally solicit for "any other comments" and give people a few more days before merging.

- Click the "Merge pull request" button at the bottom of the PR page.

- Click the "Delete branch" button.

## Guidelines for Code Style and Documentation

### Code Style

In this project we follow the [Black](https://github.com/psf/black) code formatting guidelines` (Any color you like...) This means that all code should be automatically formatted using Black and CI will fail if that's not the case.

Black should be automatically installed when you install the project with `poetry install`.

And run it manually from the root directory of your local clone with `black .`

We highly recommend installing `pre-commit`, which will take care of trailing whitespace, checking for large files, and stripping
notebook metadata (keeping output).

```bash
pre-commit install
```

And that's all you need to do from now on.

### Documentation style

BlendingToolKit follows the google format: <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>

## Credit

This file was adapted from JAX-Galsim's [CONTRIBUTING.md](https://github.com/GalSim-developers/JAX-GalSim/edit/main/CONTRIBUTING.md)
