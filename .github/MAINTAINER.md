# Maintainer

## Setup environment

For development, I recommend the following instructions to install the package:

1. Create a new `conda` environment.

2. Install `scarlet` following the conda instructions in the documentation.

3. Then run the following commands **within the new `conda` environment**:

```bash
pip install -e .
pip install -e ".[dev]"
pre-commit install
```

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
git checkout -b release/v$RELEASE origin/dev

# 6. Bump version in release branch
# edit pyproject.toml file to update the version
# also CHANGELOG.md
git add pyproject.toml
git add CHANGELOG.md
git commit -m "Version $RELEASE"

# 7. Pushed release branch to remote repository
git push --set-upstream origin release/v$RELEASE

# 8. Open a "pull request" in GitHub for team to verify the release

# 9. Checkout into main branch
git checkout main

# 10. Updated local main branch with remote copy
git pull origin main

# 11. Merged release branch into main branch
git merge release/v$RELEASE

# 12. Tagged the release point by creating a new tag
git tag -a v$RELEASE -m "Release version $RELEASE"

# 13. Pushed main branch to remote repository
git push origin main

# 14. Pushed the tags to remote repository
git push origin v$RELEASE

# 15. Checkout into dev branch
git checkout dev

# 16. Merged release branch into dev branch
git merge release/v$RELEASE

# 17. Pushed dev branch to remote repository
git push origin dev

# 18. Removed release branch from the local repository
git branch -D release/v$RELEASE

# 19. Removed release branch from the remote repository
git push origin :release/v$RELEASE


CREDIT: http://www.inanzzz.com
```
