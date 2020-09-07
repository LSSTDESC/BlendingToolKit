import warnings
from abc import ABC, abstractmethod
import numpy as np
import astropy.table


def _get_random_center_shift(num_objects, maxshift):
    """Returns random shifts in x and y coordinates between + and - max-shift
    in arcseconds.

    Args:
        num_objects(int): Number of x and y shifts to return.
    """
    dx = np.random.uniform(-maxshift, maxshift, size=num_objects)
    dy = np.random.uniform(-maxshift, maxshift, size=num_objects)
    return dx, dy


# TODO: Indicate what type of catalog is compatible with each sampling function.
class SamplingFunction(ABC):
    def __init__(self, max_number):
        """Class representing sampling functions to sample input catalog from which to draw
        blends.

        Args:
            max_number (int): maximum number of catalog entries returned from sample.
        """
        self.max_number = max_number

    @abstractmethod
    def __call__(self, catalog):
        """Returns a sample from the catalog with at most self.max_number of objects.
        Changes the 'ra' and 'dec' entries to be in arcseconds."""
        pass


class DefaultSampling(SamplingFunction):
    def __init__(
        self, max_number=2, stamp_size=24.0, maxshift=None, shifts=None, ids=None
    ):
        """
        Default sampling function used for producing blend catalogs.
        Args:
            max_number (int): Defined in parent class
            stamp_size (float):
            maxshift (float): Magnitude of maximum value of shift. If None then it
                             is set as one-tenth the stamp size. In arcseconds.
        """
        super().__init__(max_number)
        self.stamp_size = stamp_size
        self.maxshift = maxshift if maxshift else self.stamp_size / 10.0
        self.shifts = shifts
        self.ids = ids

    def __call__(self, catalog):
        """Applies default sampling to the input CatSim-like catalog and returns
        catalog with entries corresponding to a blend centered close to postage
        stamp center.

        Function selects entries from input catalog that are brighter than 25.3 mag
        in the i band. Number of objects per blend is set at a random integer
        between 1 and Args.max_number. The blend catalog is then randomly sampled
        entries from the catalog after selection cuts. The centers are randomly
        distributed within 1/10th of the stamp size. Here even though the galaxies
        are sampled from the CatSim catalog, their spatial location are not
        representative of real blends.

        Args:
            catalog: CatSim-like catalog from which to sample galaxies.

        Returns:
            Catalog with entries corresponding to one blend.
        """
        number_of_objects = np.random.randint(1, self.max_number + 1)
        (q,) = np.where(catalog["i_ab"] <= 25.3)

        if self.ids == None:
            blend_catalog = catalog[np.random.choice(q, size=number_of_objects)]
        else:
            blend_catalog = catalog[self.ids]
        blend_catalog["ra"], blend_catalog["dec"] = 0.0, 0.0
        if self.shifts == None:
            dx, dy = _get_random_center_shift(number_of_objects, self.maxshift)
        else:
            dx, dy = self.shifts
        blend_catalog["ra"] += dx
        blend_catalog["dec"] += dy

        if np.any(blend_catalog["ra"] > self.stamp_size / 2.0) or np.any(
            blend_catalog["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_catalog


class BasicSamplingFunction(SamplingFunction):
    def __init__(self, max_number=4, stamp_size=24.0, maxshift=None):
        super().__init__(max_number)
        self.stamp_size = stamp_size

    def __call__(self, catalog):
        """Samples galaxies from input catalog to make blend scene.

        Then number of galaxies in a blend are drawn from a uniform
        distribution of one up to Args.max_number. Function always selects one
        bright galaxy that is less than 24 imag. The other galaxies are selected
        from a sample with i<25.3 90% of the times and the remaining 10% with i<28.
        All galaxies must have semi-major axis is between 0.2 and 2 arcsec.
        The centers are randomly distributed within 1/30 *sqrt(N) of the postage
        stamp size, where N is the number of objects in the blend.

        Args:
            catalog: CatSim-like catalog from which to sample galaxies.

        Returns:
            Catalog with entries corresponding to one blend.
        """

        number_of_objects = np.random.randint(0, self.max_number)
        a = np.hypot(catalog["a_d"], catalog["a_b"])
        cond = (a <= 2) & (a > 0.2)
        (q_bright,) = np.where(cond & (catalog["i_ab"] <= 24))
        if np.random.random() >= 0.9:
            (q,) = np.where(cond & (catalog["i_ab"] < 28))
        else:
            (q,) = np.where(cond & (catalog["i_ab"] <= 25.3))
        blend_catalog = astropy.table.vstack(
            [
                catalog[np.random.choice(q_bright, size=1)],
                catalog[np.random.choice(q, size=number_of_objects)],
            ]
        )
        blend_catalog["ra"], blend_catalog["dec"] = 0.0, 0.0
        # keep number density of objects constant
        maxshift = self.stamp_size / 30.0 * number_of_objects ** 0.5
        dx, dy = _get_random_center_shift(number_of_objects + 1, maxshift)
        blend_catalog["ra"] += dx
        blend_catalog["dec"] += dy
        return blend_catalog


class GroupSamplingFunction(SamplingFunction):
    def __init__(
        self,
        max_number,
        wld_catalog_name,
        stamp_size,
        pixel_scale,
        shift=None,
        group_id=None,
    ):
        """Blends are defined from *groups* of galaxies from the CatSim
            catalog previously analyzed with WLD.

        Note: the pre-run WLD images are not used here. We only use the pre-run
        catalog (in i band) to identify galaxies that belong to a group.

        Args:
            max_number: Same as in SamplingFunction
            wld_catalog_name: File path to a pre-analyzed WLD Catsim catalog.
        """
        super().__init__(max_number)
        self.wld_catalog = astropy.table.Table.read(wld_catalog_name, format="fits")
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.shift = shift
        self.group_id = group_id

    def __call__(self, catalog):
        """We use self.wld_catalog created above to sample groups, but ultimately returns
        rows from catalog (by matching the corresponding galaxy ids).

        The group is centered on the middle of the postage stamp. Function only draws
        galaxies that lie within the postage stamp size.
        """

        # randomly sample a group om wld_catalog
        bool_groups = self.wld_catalog["grp_size"] >= 2
        group_ids = np.unique(self.wld_catalog["grp_id"][bool_groups])
        if self.ids == None:
            group_id = np.random.choice(group_ids, replace=False)
        else:
            group_id = self.group_id

        # get all galaxies belonging to the group.
        ids = self.wld_catalog["db_id"][self.wld_catalog["grp_id"] == group_id]
        blend_catalog = astropy.table.vstack(
            [catalog[catalog["galtileid"] == i] for i in ids]
        )

        # Set mean x and y coordinates of the group galaxies to the center of the
        # postage stamp.
        blend_catalog["ra"] -= np.mean(blend_catalog["ra"])
        blend_catalog["dec"] -= np.mean(blend_catalog["dec"])
        # convert ra dec from degrees to arcsec
        blend_catalog["ra"] *= 3600
        blend_catalog["dec"] *= 3600
        # Add small random shift so that center does not perfectly align with
        # the stamp center
        if self.shift == None:
            dx, dy = _get_random_center_shift(1, maxshift=3 * self.pixel_scale)
        else:
            dx, dy = self.shift
        blend_catalog["ra"] += dx
        blend_catalog["dec"] += dy
        # make sure galaxy centers don't lie too close to edge
        cond1 = np.abs(blend_catalog["ra"]) < self.stamp_size / 2.0 - 3
        cond2 = np.abs(blend_catalog["dec"]) < self.stamp_size / 2.0 - 3
        no_boundary = blend_catalog[cond1 & cond2]
        if len(no_boundary) == 0:
            return no_boundary
        # make sure number of galaxies in blend is less than Args.max_number
        # randomly select max_number of objects if larger.
        num = min([len(no_boundary), self.max_number])
        select = np.random.choice(range(len(no_boundary)), num, replace=False)
        return no_boundary[select]


class GroupSamplingFunctionNumbered(SamplingFunction):
    def __init__(
        self, max_number, wld_catalog_name, stamp_size, pixel_scale, shift=None
    ):
        """Blends are defined from *groups* of galaxies from a CatSim-like
        catalog previously analyzed with WLD.

        This function has an extra attribute group_id_count which tracks the
        group id returned. Each time the generator is called, 1 gets added to the
        count. If the count is larger than the number of groups input,
        the generator is forced to exit.

        Note: the pre-run WLD images are not used here. We only use the pre-run
        catalog to identify galaxies that belong to a group.

        Args:
            max_number: Same as SamplingFunction
            wld_catalog_name: Same as GroupSamplingFunction
        """
        super().__init__(max_number)
        self.wld_catalog = astropy.table.Table.read(wld_catalog_name, format="fits")
        self.stamp_size = stamp_size
        self.pixel_scale = pixel_scale
        self.group_id_count = 0
        self.shift = shift

    def __call__(self, catalog):
        """The group is centered on the middle of the postage stamp.
        This function only returns galaxies whose centers lie within 1 arcsec the
        postage stamp edge, which may cause the number of galaxies in the blend to
        be smaller than the group size.
        """

        # randomly sample a group.
        group_ids = np.unique(
            self.wld_catalog["grp_id"][
                (self.wld_catalog["grp_size"] >= 2)
                & (self.wld_catalog["grp_size"] <= self.max_number)
            ]
        )
        if self.group_id_count >= len(group_ids):
            message = "group_id_count is larger than number of groups input"
            raise GeneratorExit(message)

        group_id = group_ids[self.group_id_count]
        self.group_id_count += 1
        # get all galaxies belonging to the group.
        # make sure some group or galaxy was not repeated in wld_catalog
        ids = np.unique(
            self.wld_catalog["db_id"][self.wld_catalog["grp_id"] == group_id]
        )
        blend_catalog = astropy.table.vstack(
            [catalog[catalog["galtileid"] == i] for i in ids]
        )
        # Set mean x and y coordinates of the group galaxies to the center of the
        # postage stamp.
        blend_catalog["ra"] -= np.mean(blend_catalog["ra"])
        blend_catalog["dec"] -= np.mean(blend_catalog["dec"])
        # convert ra dec from degrees to arcsec
        blend_catalog["ra"] *= 3600
        blend_catalog["dec"] *= 3600
        # Add small random shift so that center does not perfectly align with stamp
        # center
        if self.shift == None:
            dx, dy = _get_random_center_shift(1, maxshift=5 * self.pixel_scale)
        else:
            dx, dy = self.shift
        blend_catalog["ra"] += dx
        blend_catalog["dec"] += dy
        # make sure galaxy centers don't lie too close to edge
        cond1 = np.abs(blend_catalog["ra"]) < self.stamp_size / 2.0 - 1
        cond2 = np.abs(blend_catalog["dec"]) < self.stamp_size / 2.0 - 1
        no_boundary = blend_catalog[cond1 & cond2]
        message = (
            "Number of galaxies greater than max number of objects per"
            f"blend. Found {len(no_boundary)}, expected <= {self.max_number}"
        )
        assert len(no_boundary) <= self.max_number, message
        return no_boundary
