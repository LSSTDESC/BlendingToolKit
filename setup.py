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
        "numpy>=1.12",
        "astropy>=2.0",
        "scipy>=1.2.0",
        "matplotlib>=3.3.3",
        "sep>=1.1.1",
        "scikit-image>=0.18.0",
        "galsim>=2.2.5",
    ],
    python_requires=">=3.7",
)
