"""Class which stores all relevant data for blends."""
import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Union

import galsim
import numpy as np
from astropy.table import Table

from btk.survey import Survey, make_wcs


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
        """Save the batch to disk.

        Args:
            path (str): Path to save the batch to.
            batch_number (int): Number of the batch.
        """
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, f"blend_images_{batch_number}.npy"), self.blend_images)
        np.save(os.path.join(path, f"isolated_images_{batch_number}.npy"), self.isolated_images)

        with open(os.path.join(path, f"psf_{batch_number}.pickle"), "wb") as f:
            pickle.dump(self.psf, f)

        with open(os.path.join(path, f"catalog_list_{batch_number}.pickle"), "wb") as f:
            pickle.dump(self.catalog_list, f)

        # save general info about blend
        with open(os.path.join(path, "blend.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "batch_size": self.batch_size,
                    "max_n_sources": self.max_n_sources,
                    "stamp_size": self.stamp_size,
                    "survey_name": self.survey.name,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, batch_number: int = 0):
        """Load the batch from disk.

        Args:
            path (str): Path to load the batch from.
            batch_number (int): Number of the batch.
        """
        # load general infrom about blend
        with open(os.path.join(path, "blend.json"), "r", encoding="utf-8") as f:
            blend_info = json.load(f)
        batch_size = blend_info["batch_size"]
        max_n_sources = blend_info["max_n_sources"]
        stamp_size = blend_info["stamp_size"]
        survey_name = blend_info["survey_name"]

        blend_images = np.load(os.path.join(path, f"blend_images_{batch_number}.npy"))
        isolated_images = np.load(os.path.join(path, f"isolated_images_{batch_number}.npy"))

        # load psfs
        with open(os.path.join(path, f"psf_{batch_number}.pickle"), "rb") as f:
            psf = pickle.load(f)

        # load catalog
        with open(os.path.join(path, f"catalog_list_{batch_number}.pickle"), "rb") as f:
            catalog_list = pickle.load(f)

        return cls(
            batch_size,
            max_n_sources,
            stamp_size,
            survey_name,
            blend_images,
            isolated_images,
            catalog_list,
            psf,
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
        """Save batch of measure results to disk."""
        save_dir = os.path.join(path, str(batch_number))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            if self.segmentation is not None:
                np.save(os.path.join(save_dir, "segmentation"), self.segmentation)
            if self.deblended_images is not None:
                np.save(os.path.join(save_dir, "deblended_images"), self.deblended_images)
            with open(os.path.join(save_dir, "catalog_list.pickle"), "wb") as f:
                pickle.dump(self.catalog_list, f)

        # save general info about class
        with open(os.path.join(path, "meas.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "batch_size": self.batch_size,
                    "max_n_sources": self.max_n_sources,
                    "image_size": self.image_size,
                    "n_bands": self.n_bands,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, batch_number: int = 0):
        """Load batch of measure results from disk."""
        load_dir = os.path.join(path, str(batch_number))
        with open(os.path.join(path, "meas.json"), "r", encoding="utf-8") as f:
            meas_config = json.load(f)

        with open(os.path.join(load_dir, "catalog_list.pickle"), "rb") as f:
            catalog_list = pickle.load(f)
        segmentation, deblended_images = None, None
        if os.path.exists(os.path.join(load_dir, "segmentation.npy")):
            segmentation = np.load(os.path.join(load_dir, "segmentation.npy"))
        if os.path.exists(os.path.join(load_dir, "deblended_images.npy")):
            deblended_images = np.load(os.path.join(load_dir, "deblended_images.npy"))
        return cls(
            catalog_list=catalog_list,
            segmentation=segmentation,
            deblended_images=deblended_images,
            **meas_config,
        )
