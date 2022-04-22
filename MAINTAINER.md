# Setup environment

1. We use [poetry](https://python-poetry.org) as python package manager for BTK. It guarantees all developers are sharing the same python environment, makes it really easy to update dependencies, and publish to [pypi](https://pypi.org). Given some of the complications with installing `galsim` via `poetry`, we follow a hybrid approach of `conda`+`poetry`.

2. It is recommended to create a `conda` virtual environment (using `python3.8`) from scratch and use it to install all required dependencies. After having installed `conda`, please follow the following series of steps:

```
# enter to the local repo
cd BlendingToolKit

# create virtual environment.
conda create -n btk python=3.8
conda activate btk

# make sure you can import galsim after this step before moving on.
conda install fftw
conda install -c conda-forge eigen
conda install -c conda-forge galsim

# poetry allows us to install all dependencies directly from pyproject.toml (except galsim!)
# poetry reuses the current "btk" virtual environment from conda.
conda install -c conda-forge poetry==1.1.10
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

# Updating packages

1. If any of the dependencies requires an update, you can simply run `poetry update` inside your local repo to automatically update and install them. Feel free to push the changes of the `pyproject.toml` or `poetry.lock` file to the PR you are working on.

2. You might also want to update the `requirements.txt` file every now and then:

```
poetry export -o requirements.txt --without-hashes --dev
```

ideally everytime you run `poetry update`.

# Making new Releases

```
# 1. Checked out into develop branch
git checkout develop

# 2. Fetched all remote updates
git remote update

# 3. Update local develop branch with remote copy
git pull origin develop

# 4. Created a release branch that tracks origin/develop
git checkout -b release/0.1.0 origin/develop

# 5. Pushed release branch to remote repository
git push origin release/0.1.0

# 6. Opened a "pull request" in GitHub for team to verify the release

# 7. Checkout into main branch
git checkout main

# 8. Updated local main branch with remote copy
git pull origin main

# 9. Merged release branch into main branch
git merge release/0.1.0

# 10. Tagged the release point by creating a new tag
git tag -a 0.1.0 -m 'Create release tag 0.1.0'

# 11. Pushed main branch to remote repository
git push origin main

# 12. Pushed the tags to remote repository
git push origin --tags

# 13. Checkout into develop branch
git checkout develop

# 14. Merged release branch into develop branch
git merge release/0.1.0

# 15. Pushed develop branch to remote repository
git push origin develop

# 16. Removed release branch from the local repository
git branch -D release/0.1.0

# 17. Removed release branch from the remote repository
git push origin :release/0.1.0


CREDIT: http://www.inanzzz.com
```
