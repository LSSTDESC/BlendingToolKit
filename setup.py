from setuptools import setup

setup(
    name="btk",
    version="0.1",
    description="Toolkit for on the fly generation of blend images.",
    long_description="Weak lensing fast simulations and analysis for the " "LSST DESC",
    author="btk developers",
    author_email="sowmyak@stanford.edu",
    url="https://github.com/LSSTDESC/BlendingToolKit",
    packages=["btk"],
    # scripts = [ ],
    # include_package_data=True,
    # zip_safe=False,
    install_requires=[
        "lmfit",
        "fitsio",
    ],
    license="MIT",
)
