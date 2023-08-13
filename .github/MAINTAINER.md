# Maintainer

## Setup environment

1. We use [poetry](https://python-poetry.org/docs/) as python package manager for BTK. It guarantees all developers are sharing the same python environment, makes it really easy to update dependencies, and publish to [pypi](https://pypi.org).

2. For development, it is recommended to create a `poetry` virtual environment from scratch and use it to install all required dependencies. Please follow the following series of steps:

```bash
# enter to the local repo
cd BlendingToolKit

# install poetry
curl -sSL https://install.python-poetry.org | python3 -

# install all python dependencies from pyproject.toml file
# and create lock file
poetry update

# activate pdm environment
poetry shell

# install the git hook scripts
pre-commit install
```

Remember that `galsim` has additional dependencies that you might need to install prior to running `poetry update`. Please follow the instructions [here](https://galsim-developers.github.io/GalSim/_build/html/install.html).

One workaround in case of problems is to setup a `conda` environment, install `galsim`, install `poetry` with `conda`, and then run `poetry update`.

## Updating packages

If any of the dependencies requires an update, you can simply run `poetry update` inside your local repo to automatically update and install them. Feel free to push the changes of the `poetry.lock` file to the PR you are working on.

## Making new Releases

**Note:** Remember to update the latest package version in the `requirements.txt` file.

```bash
# 0. Create release tag
export RELEASE=XX.YY.ZZ

# 1. Checked out into dev branch
git checkout dev

# 2. Fetched all remote updates
git remote update

# 3. Update local dev branch with remote copy
git pull origin dev

# 4. Created a release branch that tracks origin/dev
git checkout -b release/$RELEASE origin/dev

# 5. Bump version in release branch
# edit pyproject.toml file to update the version
poetry version $RELEASE
git add pyproject.toml
git commit -m "Bump version"

# 6. Pushed release branch to remote repository
git push --set-upstream origin release/$RELEASE

# 7. Open a "pull request" in GitHub for team to verify the release

# 8. Checkout into main branch
git checkout main

# 9. Updated local main branch with remote copy
git pull origin main

# 10. Merged release branch into main branch
git merge release/$RELEASE

# 11. Tagged the release point by creating a new tag
git tag -a $RELEASE -m "Create release tag $RELEASE"

# 12. Pushed main branch to remote repository
git push origin main

# 13. Pushed the tags to remote repository
git push origin --tags

# 14. Checkout into dev branch
git checkout dev

# 15. Merged release branch into dev branch
git merge release/$RELEASE

# 16. Pushed dev branch to remote repository
git push origin dev

# 17. Removed release branch from the local repository
git branch -D release/$RELEASE

# 18. Removed release branch from the remote repository
git push origin :release/$RELEASE


CREDIT: http://www.inanzzz.com
```
