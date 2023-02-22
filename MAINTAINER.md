# Maintainer

## Setup environment

1. We use [pdm](https://pdm.fming.dev/latest/) as python package manager for BTK. It guarantees all developers are sharing the same python environment, makes it really easy to update dependencies, and publish to [pypi](https://pypi.org).

2. It is recommended to create a `pdm` virtual environment from scratch and use it to install all required dependencies. Please follow the following series of steps:

```bash
# enter to the local repo
cd BlendingToolKit

# install pdm in mac os
brew install pdm

# install all python dependencies from pyproject.toml file
# and create lock file
pdm update

# activate pdm environment
eval $(pdm venv activate in-project)

# install the git hook scripts
pre-commit install
```

Remember that `galsim` has additional dependencies that you might need to install. Please follow the instructions [here](https://galsim-developers.github.io/GalSim/_build/html/install.html).

## Updating packages

1. If any of the dependencies requires an update, you can simply run `pdm update` inside your local repo to automatically update and install them. Feel free to push the changes of the `pyproject.toml` or `pdm.lock` file to the PR you are working on.

2. You might also want to update the `requirements.txt` anytime you edit the `pyproject.toml` file.

```bash
pdm export -f requirements --without-hashes --pyproject --prod > requirements.txt
```

## Making new Releases

**Note:** Remember to update the latest package version in the `requirements.txt` file.

```bash
# 0. Create release tag
export RELEASE=XX.YY.ZZ

# 1. Checked out into develop branch
git checkout develop

# 2. Fetched all remote updates
git remote update

# 3. Update local develop branch with remote copy
git pull origin develop

# 4. Created a release branch that tracks origin/develop
git checkout -b release/$RELEASE origin/develop

# 5. Bump version in release branch
# edit pyproject.toml file to update the version
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

# 14. Checkout into develop branch
git checkout develop

# 15. Merged release branch into develop branch
git merge release/$RELEASE

# 16. Pushed develop branch to remote repository
git push origin develop

# 17. Removed release branch from the local repository
git branch -D release/$RELEASE

# 18. Removed release branch from the remote repository
git push origin :release/$RELEASE


CREDIT: http://www.inanzzz.com
```
