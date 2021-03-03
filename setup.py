import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blending_toolkit",
    version="0.0.1-alpha.1",
    author="Ismael Mendoza",
    author_email="imendoza@umich.edu",
    description="Toolkit for on the fly generation of blend images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LSSTDESC/BlendingToolKit",
    project_urls={
        "Bug Tracker": "https://github.com/LSSTDESC/BlendingToolKit/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=["btk", "btk.*"]),
    install_requires=[
        "numpy",
        "astropy",
        "scipy",
        "matplotlib",
        "sep",
        "scikit-image",
        "galsim",
    ],
    python_requires=">=3.7",
)
