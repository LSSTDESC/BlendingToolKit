"""Contains class `BlendGenerator` to combine entries from a given catalog into blends."""


from btk.catalog import Catalog
from btk.sampling_functions import SamplingFunction


class BlendGenerator:
    """Class that uses a catalog and a sampling function to return blend information in batches."""

    def __init__(
        self,
        catalog: Catalog,
        sampling_function: SamplingFunction,
        batch_size: int = 8,
        verbose: bool = False,
    ):
        """Initializes the BlendGenerator.

        Args:
            catalog: BTK Catalog object
            sampling_function: An object that return samples from catalog.
            batch_size: Size of batches returned. (Default: 8)
            verbose: Whether to print additional information.
        """
        self.catalog = catalog
        self.batch_size = batch_size
        self.verbose = verbose
        self.sampling_function = sampling_function

        if not hasattr(sampling_function, "max_number"):
            raise AttributeError(
                "Please change your custom sampling function to have an attribute 'max_number'."
            )

        self.max_number = self.sampling_function.max_number
        self.min_number = self.sampling_function.min_number

    def __iter__(self):
        """Returns an iterable which is the object itself."""
        return self

    def _check_n_sources(self, blend_table):
        if len(blend_table) > self.max_number:
            raise ValueError(
                f"Number of objects per blend must be "
                f"less than max_number: {len(blend_table)} <= {self.max_number}"
            )
        if len(blend_table) < self.min_number:
            raise ValueError(
                f"Number of objects per blend must be "
                f"greater than min_number: {len(blend_table)} >= {self.min_number}"
            )

    def __next__(self):
        """Generates a list of blend tables of len batch_size.

        Each blend table has entries numbered between `min_number` and `max_number`, corresponding
        to overlapping objects in the blend.

        Returns:
            blend_tables (list): a list of astropy tables, each one corresponding to a blend.
        """
        blend_tables = []
        for _ in range(self.batch_size):
            blend_table = self.sampling_function(self.catalog.table)
            self._check_n_sources(blend_table)
            blend_tables.append(blend_table)
        return blend_tables
