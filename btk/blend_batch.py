"""Class which stores all relevant data for blends."""
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import galsim
import h5py
import numpy as np
from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5
from astropy.table import Table

from btk.survey import Survey, get_surveys, make_wcs


@dataclass
class BlendBatch:
    """Class which stores all relevant data for blends in a single survey."""

    batch_size: int
    max_n_sources: int
    stamp_size: int
    survey: Survey
    blend_images: np.ndarray
    isolated_images: np.ndarray
    catalog_list: List[Table]
    psf: List[galsim.GSObject]  # each element corresponds to each band

    def __post_init__(self):
        """Checks that the data is of the right shape."""
        self.wcs = self._get_wcs()
        self.image_size = self._get_image_size()
        n_bands = len(self.survey.available_filters)
        b1, c1, ps11, ps12 = self.blend_images.shape
        b2, n, c2, ps21, ps22 = self.isolated_images.shape
        assert b1 == b2 == len(self.catalog_list) == self.batch_size
        assert c1 == c2 == n_bands
        assert n == self.max_n_sources
        assert ps11 == ps12 == ps21 == ps22 == self._get_image_size()
        assert self.isolated_images.min() >= 0

    def _get_image_size(self) -> int:
        """Returns the size of the stamps in pixels."""
        pixel_scale = self.survey.pixel_scale.to_value("arcsec")
        return int(self.stamp_size / pixel_scale)

    def _get_wcs(self):
        """Returns the wcs of the stamps."""
        pix_stamp_size = self._get_image_size()
        pixel_scale = self.survey.pixel_scale.to_value("arcsec")
        return make_wcs(pixel_scale, (pix_stamp_size, pix_stamp_size))

    def get_numpy_psf(self):
        """Returns the psf as a numpy array."""
        return np.array(
            [psf.drawImage(nx=self.image_size, ny=self.image_size).array for psf in self.psf]
        )

    def __repr__(self) -> str:
        """Return string representation of class."""
        string = self.__class__.__name__ + f"(survey_name={self.survey.name}, "
        string += "\n\t blend_images: np.ndarray, shape " + str(list(self.blend_images.shape))
        string += "\n\t isolated_images: np.ndarray, shape " + str(list(self.isolated_images.shape))
        string += (
            "\n\t catalog_list: list of " + str(Table) + ", size " + str(len(self.catalog_list))
        )
        string += "\n\t psfs: list of " + str(galsim.GSObject) + ", size " + str(len(self.psf))
        string += "\n\t wcs: " + str(type(self.wcs)) + ")"
        return string

    def save(self, path: str, batch_number: int = 0):
        """Save the batch to disk using hdf5 format.

        Args:
            path (str): Path to save the batch to.
            batch_number (int): Number of the batch.
        """
        fpath = os.path.join(path, f"blend_{batch_number}.hdf5")

        with h5py.File(fpath, "w") as f:
            # save blend and isolated images
            f.create_dataset("blend_images", data=self.blend_images)
            f.create_dataset("isolated_images", data=self.isolated_images)

            # save psfs
            # first convert psfs to numpy array
            psf_array = self.get_numpy_psf()
            f.create_dataset("psf", data=psf_array)

            # save catalog using astropy functions
            # (this is faster than saving as numpy array)
            for ii, catalog in enumerate(self.catalog_list):
                write_table_hdf5(catalog, f, path=f"catalog_list/{ii}")

            # save general info about blend
            f.attrs["batch_size"] = self.batch_size
            f.attrs["max_n_sources"] = self.max_n_sources
            f.attrs["stamp_size"] = self.stamp_size
            f.attrs["survey_name"] = self.survey.name

    @classmethod
    def load(cls, path: str, batch_number: int = 0):
        """Load the batch from hdf5 format.

        Args:
            path (str): Path to load the batch from.
            batch_number (int): Number of the batch.
        """
        # file path
        fpath = os.path.join(path, f"blend_{batch_number}.hdf5")

        # open file
        with h5py.File(fpath, "r") as f:
            # load blend and isolated images
            blend_images = f["blend_images"][:]
            isolated_images = f["isolated_images"][:]

            # load psfs
            psf_list = [galsim.Image(psf) for psf in f["psf"][:]]

            # load catalog
            catalog_list = []
            for ii in range(f.attrs["batch_size"]):
                catalog_list.append(read_table_hdf5(f, path=f"catalog_list/{ii}"))

            # load general info about blend
            batch_size = f.attrs["batch_size"]
            max_n_sources = f.attrs["max_n_sources"]
            stamp_size = f.attrs["stamp_size"]
            survey_name = f.attrs["survey_name"]

        # create survey
        survey = get_surveys(survey_name)

        # create class
        return cls(
            batch_size=batch_size,
            max_n_sources=max_n_sources,
            stamp_size=stamp_size,
            survey=survey,
            blend_images=blend_images,
            isolated_images=isolated_images,
            catalog_list=catalog_list,
            psf=psf_list,
        )


class MultiResolutionBlendBatch:
    """Class which stores blend information for multiple surveys."""

    def __init__(self, blend_batch_list: List[BlendBatch]):
        """Initialise the class and input format."""
        assert len(blend_batch_list) > 0
        self.batch_size = blend_batch_list[0].batch_size
        self.max_n_sources = blend_batch_list[0].max_n_sources
        self.stamp_size = blend_batch_list[0].stamp_size
        self.survey_names = [blend_batch.survey_name for blend_batch in blend_batch_list]
        for blend_batch in blend_batch_list:
            assert isinstance(blend_batch, BlendBatch)
        self.results = {blend_batch.survey_name: blend_batch for blend_batch in blend_batch_list}

    def __getitem__(self, item: Union[str, int, slice]):
        """Return SurveyBatch for a given survey name or index."""
        if isinstance(item, (int, slice)):
            return list(self.results.values())[item]
        return self.results[item]

    def __repr__(self):
        """Return string representation of class."""
        string = (
            f"SurveysBatch(batch_size = {self.batch_size}, "
            f"max_n_sources = {self.max_n_sources}, stamp_size = {self.stamp_size}), containing:"
        )
        for _, blend_batch in self.results.items():
            string += "\n" + blend_batch.__repr__()
        return string

    def save(self, path: str, batch_number: int = 0):
        """Save blend results into path."""
        for survey_name, blend_batch in self.results.items():
            survey_directory = os.path.join(path, str(batch_number), survey_name)
            if not os.path.exists(survey_directory):
                os.makedirs(survey_directory)
            blend_batch.save(survey_directory, batch_number)

    @classmethod
    def load(cls, path: str, batch_number: int = 0):
        """Load blend results from path."""
        blend_batch_list = []
        for survey_name in os.listdir(os.path.join(path, str(batch_number))):
            blend_batch_list.append(
                BlendBatch.load(os.path.join(path, str(batch_number), survey_name), batch_number)
            )
        return cls(blend_batch_list)


@dataclass
class DeblendExample:
    """Class that validates the deblending results for a single blend."""

    max_n_sources: int
    catalog: Table
    n_bands: Optional[int] = None
    image_size: Optional[int] = None
    segmentation: Optional[np.ndarray] = None
    deblended_images: Optional[np.ndarray] = None
    extra_data: Optional[dict] = None

    def __post_init__(self) -> None:
        """Performs validation of the measured example."""
        self.catalog = self._validate_catalog(self.catalog)
        self.segmentation = self._validate_segmentation(self.segmentation)
        self.deblended_images = self._validate_deblended_images(self.deblended_images)

    def _validate_catalog(self, catalog: Table):
        if not ("ra" in catalog.colnames and "dec" in catalog.colnames):
            raise ValueError(
                "The output catalog of at least one of your measurement functions does"
                "not contain the mandatory 'ra' and 'dec' columns"
            )
        if not len(catalog) <= self.max_n_sources:
            raise ValueError(
                "The predicted catalog of at least one of your deblended images "
                "contains more sources than the maximum number of sources specified."
            )
        return catalog

    def _validate_segmentation(self, segmentation):
        if segmentation is not None:
            if self.image_size is None or self.n_bands is None:
                raise ValueError("`image_size` must be specified if segmentation is provided")
            if segmentation.shape != (self.max_n_sources, self.image_size, self.image_size):
                raise ValueError(
                    "The predicted segmentation of at least one of your deblended images "
                    "has the wrong shape. It should be `(max_n_sources, image_size, image_size)`."
                )
            if segmentation.min() < 0 or segmentation.max() > 1:
                raise ValueError(
                    "The predicted segmentation of at least one of your deblended images "
                    "has values outside the range [0, 1]."
                )
        return segmentation

    def _validate_deblended_images(self, deblended_images):
        if deblended_images is not None:
            if self.image_size is None or self.n_bands is None:
                raise ValueError(
                    "`image_size` and `n_bands` must be specified if deblended_images is provided"
                )
            deblended_shape = (
                self.max_n_sources,
                self.n_bands,
                self.image_size,
                self.image_size,
            )
            if deblended_images.shape != deblended_shape:
                raise ValueError(
                    "The predicted deblended_images of at least one of your deblended images "
                    f"has the wrong shape. It should be {deblended_shape}."
                )
            if deblended_images.min() < 0:
                raise ValueError(
                    "The predicted deblended_images of at least one of your "
                    "deblended images has negative values which is unphysical."
                )
        return deblended_images

    def __repr__(self):
        """Return string representation of class."""
        string = (
            f"DeblendExample(max_n_sources = {self.max_n_sources}, "
            f"n_bands = {self.n_bands}, "
            f"image_size = {self.image_size})" + ", \n containing: \n"
        )
        string += "\tcatalog: " + str(Table)

        if self.segmentation is not None:
            string += (
                "\n\tsegmentation: "
                + str(np.ndarray)
                + ", shape "
                + str(list(self.segmentation.shape))
            )
        else:
            string += "\n\tsegmentation: None"

        if self.deblended_images is not None:
            string += "\n\tdeblended_images: " + str(np.ndarray) + ", shape "
            string += str(list(self.deblended_images.shape))
        else:
            string += "\n\tdeblended_images: None"
        return string


@dataclass
class DeblendBatch:
    """Class that validates the deblending results for a batch of images in a single survey."""

    batch_size: int
    max_n_sources: int
    catalog_list: List[Table]
    n_bands: Optional[int] = None
    image_size: Optional[int] = None
    segmentation: Optional[np.ndarray] = None
    deblended_images: Optional[np.ndarray] = None
    extra_data: Optional[List[dict]] = None

    def __post_init__(self) -> None:
        """Run after dataclass init."""
        self.catalog_list = self._validate_catalog(self.catalog_list)
        self.segmentation = self._validate_segmentation(self.segmentation)
        self.deblended_images = self._validate_deblended_images(self.deblended_images)

    def _validate_catalog(self, catalog_list: List[Table]):
        if not isinstance(catalog_list, list):
            raise TypeError(
                "Catalog must be a list of 'astropy.table.Table' for each image in the batch"
            )
        assert len(catalog_list) == self.batch_size
        for catalog in catalog_list:
            if not ("ra" in catalog.colnames and "dec" in catalog.colnames):
                raise ValueError(
                    "The output catalog of at least one of your measurement functions does"
                    "not contain the mandatory 'ra' and 'dec' columns"
                )
            if not len(catalog) <= self.max_n_sources:
                raise ValueError(
                    "The predicted catalog of at least one of your deblended images "
                    "contains more sources than the maximum number of sources specified."
                )
        return catalog_list

    def _validate_segmentation(self, segmentation: Optional[np.ndarray] = None) -> np.ndarray:
        if segmentation is not None:
            if self.image_size is None:
                raise ValueError("`image_size` must be specified if segmentation is provided")
            assert segmentation.shape == (
                self.batch_size,
                self.max_n_sources,
                self.image_size,
                self.image_size,
            )
            assert segmentation.min() >= 0 and segmentation.max() <= 1
        return segmentation

    def _validate_deblended_images(
        self, deblended_images: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if deblended_images is not None:
            if self.image_size is None or self.n_bands is None:
                raise ValueError(
                    "`image_size` and `n_bands` must be specified if deblended_images is provided"
                )
            assert deblended_images.shape == (
                self.batch_size,
                self.max_n_sources,
                self.n_bands,
                self.image_size,
                self.image_size,
            )

            if deblended_images.min() < 0:
                raise ValueError(
                    "The predicted deblended_images of at least one of your "
                    "deblended images has negative values which is unphysical."
                )
        return deblended_images

    def __repr__(self) -> str:
        """Return string representation of class."""
        string = (
            f"DeblendBatch(batch_size = {self.batch_size}, "
            f"max_n_sources = {self.max_n_sources} "
            f"n_bands = {self.n_bands}, "
            f"image_size = {self.image_size})" + ", containing: \n"
        )
        string += "\tcatalog_list: list of " + str(Table) + ", size " + str(len(self.catalog_list))

        if self.segmentation is not None:
            string += (
                "\n\tsegmentation: "
                + str(np.ndarray)
                + ", shape "
                + str(list(self.segmentation.shape))
            )
        else:
            string += "\n\tsegmentation: None"

        if self.deblended_images is not None:
            string += (
                "\n\tdeblended_images: "
                + str(np.ndarray)
                + ", shape "
                + str(list(self.deblended_images.shape))
            )
        else:
            string += "\n\tdeblended_images: None"
        return string

    def save(self, path: str, batch_number: int = 0):
        """Save batch of measure results to disk in hdf5 format."""
        fpath = os.path.join(path, f"deblend_{batch_number}.hdf5")
        with h5py.File(fpath, "w") as f:
            # save catalog with astropy hdf5 functions
            for ii, catalog in enumerate(self.catalog_list):
                write_table_hdf5(catalog, f, path=f"catalog_list/{ii}")

            # save segmentation
            if self.segmentation is not None:
                f.create_dataset("segmentation", data=self.segmentation)

            # save deblended images
            if self.deblended_images is not None:
                f.create_dataset("deblended_images", data=self.deblended_images)

            # save general info about class
            f.attrs["batch_size"] = self.batch_size
            f.attrs["max_n_sources"] = self.max_n_sources
            f.attrs["image_size"] = self.image_size
            f.attrs["n_bands"] = self.n_bands

    @classmethod
    def load(cls, path: str, batch_number: int = 0):
        """Load batch of measure results from hdf5 file in disk."""
        fpath = os.path.join(path, f"deblend_{batch_number}.hdf5")

        # open file
        with h5py.File(fpath, "r") as f:
            # load catalog with astropy hdf5 functions
            catalog_list = []
            for ii in range(f.attrs["batch_size"]):
                catalog_list.append(read_table_hdf5(f, path=f"catalog_list/{ii}"))

            # load segmentation
            if "segmentation" in f.keys():
                segmentation = f["segmentation"][:]
            else:
                segmentation = None

            # load deblended images
            if "deblended_images" in f.keys():
                deblended_images = f["deblended_images"][:]
            else:
                deblended_images = None

            # load general info about blend
            batch_size = f.attrs["batch_size"]
            max_n_sources = f.attrs["max_n_sources"]
            image_size = f.attrs["image_size"]
            n_bands = f.attrs["n_bands"]

        # create class
        return cls(
            batch_size=batch_size,
            max_n_sources=max_n_sources,
            catalog_list=catalog_list,
            n_bands=n_bands,
            image_size=image_size,
            segmentation=segmentation,
            deblended_images=deblended_images,
        )
