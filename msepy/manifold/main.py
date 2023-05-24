# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/30/2023 7:02 PM
"""
from importlib import import_module
from tools.frozen import Frozen
from src.config import get_embedding_space_dim
from msepy.manifold.regions.main import MseManifoldRegions
from msepy.manifold.coordinate_transformation import MsePyManifoldsCoordinateTransformation
from msepy.manifold.visualize.main import MsePyManifoldVisualize

from msepy.manifold.regions.standard.main import MsePyManifoldStandardRegion
from msepy.manifold.regions.standard.ct import MsePyStandardRegionCoordinateTransformation

from msepy.manifold.regions.boundary.main import MsePyManifoldBoundaryRegion


def config(mf, arg, **kwargs):
    """"""
    assert mf.__class__ is MsePyManifold, f"I can only config MseManifold instance. Now I get {mf}."
    assert mf._regions is None, f"manifold {mf} already have region configurations. Change it may lead to error."

    if isinstance(arg, str):  # use predefined mappings
        predefined_path = '.'.join(str(MsePyManifold).split(' ')[1][1:-2].split('.')[:-2]) + \
                          '.predefined.' + arg
        _module = import_module(predefined_path)
        region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict = getattr(_module, arg)(mf, **kwargs)

        mf._parse_regions_from_region_map(
            0,
            region_map,
            mapping_dict,
            Jacobian_matrix_dict,
            mtype_dict
        )
        assert mf.regions.map is not None, f"predefined manifold only config manifold with region map."

    else:
        raise NotImplementedError()

    assert mf.regions._regions != dict(), f"we need to set regions for the manifold by this `config` function."


class MsePyManifold(Frozen):
    """"""

    def __init__(self, abstract_manifold):
        """"""
        self._abstract = abstract_manifold
        self._ct = None
        self._regions = None
        self._visualize = None
        self._freeze()

    def _parse_regions_from_region_map(
            self,
            region_map_type,
            region_map,
            mapping_dict,
            Jacobian_matrix_dict,
            mtype_dict
    ):
        """Will set `regions._map` and `region._regions` and `regions._map_type`.

        And if it has a boundary, will parse the boundary manifold as well.
        """
        assert self._regions is None, f"Change regions will be dangerous!"
        self._regions = MseManifoldRegions(self)  # initialize the regions.
        for i in region_map:

            mapping = mapping_dict[i]

            if Jacobian_matrix_dict is None:
                Jacobian_matrix = None
            else:
                Jacobian_matrix = Jacobian_matrix_dict[i]

            if mtype_dict is None:
                mtype = None
            else:
                mtype = mtype_dict[i]

            rct = MsePyStandardRegionCoordinateTransformation(mapping, Jacobian_matrix, mtype)
            region = MsePyManifoldStandardRegion(self.regions, i, rct)
            self.regions._regions[i] = region

        self.regions._map = region_map  # ***
        self._regions._map_type = region_map_type
        assert self.abstract._is_periodic is self.regions._is_periodic(), f"Periodicity does not match."

        boundary_manifold = self.abstract.boundary()
        from src.manifold import NullManifold
        if boundary_manifold.__class__ is NullManifold:
            assert self.abstract._is_periodic, f"A manifold has null boundary must be periodic."
        else:
            from msepy.main import base
            manifolds = base['manifolds']
            the_msepy_boundary_manifold = None
            for sym in manifolds:
                msepy_manifold = manifolds[sym]
                if msepy_manifold.abstract is boundary_manifold:

                    the_msepy_boundary_manifold = msepy_manifold
                    break

                else:
                    pass

            if the_msepy_boundary_manifold is None:
                # did not find the msepy boundary manifold, which means it is not used at all, just skip then.
                pass
            else:
                boundary_dict = dict()
                r_map = self.regions.map
                for i in self.regions:
                    map_i = r_map[i]
                    boundary_dict[i] = [0 for _ in range(len(map_i))]
                    for j, mp in enumerate(map_i):
                        if mp is None:
                            boundary_dict[i][j] = 1

                the_msepy_boundary_manifold._parse_regions_from_boundary_dict(self, boundary_dict)

        self.regions._check_map()

    def _parse_regions_from_boundary_dict(self, base_manifold, boundary_dict):
        """

        Parameters
        ----------
        base_manifold
        boundary_dict: dict
            {
                0: [0, 0, 1, 0, 1, ...],
                1: [1, 0, 0, ...],
                ...
            }

            It means this boundary manifold covers the y- face of region #0, z- face of region #0, ..., and
            x- face of region #1, ..., and ...

        Returns
        -------

        """
        assert self._regions is None, f"regions must be empty! Changing it is dangerous."
        # --- first check whether the boundary_dict is valid ----------------------
        assert base_manifold.regions.is_structured(), \
            f"cannot parse regions from boundary dict based on structured manifold regions (have `map`)."
        base_region_maps = base_manifold.regions.map
        for i in boundary_dict:
            assert i in base_region_maps, f"base manifold has no region #{i} at all!"
            map_i = base_region_maps[i]
            boundaries = boundary_dict[i]
            assert len(boundaries) == len(map_i), f"boundary value shape wrong for region #{i}."
            for j, bd in enumerate(boundaries):
                if bd == 0:
                    pass
                else:
                    assert bd == 1, f"boundary dict can only have 0 or 1 in the list for each base region."
                    mp = map_i[j]
                    assert mp is None, f"{j}th side of region #{i} is not a domain boundary!"

        # ------ boundary dict is OK, let's move on --------------
        self._regions = MseManifoldRegions(self)

        regions = self.regions
        regions._map_type = 1  # type 1
        regions._base_regions = base_manifold.regions
        regions._map = boundary_dict

        base_regions = base_manifold.regions
        num_regions = 0
        for i in boundary_dict:
            boundaries = boundary_dict[i]
            for j, bd in enumerate(boundaries):
                if bd == 1:
                    regions._regions[num_regions] = MsePyManifoldBoundaryRegion(
                        num_regions,
                        regions,
                        base_regions[i],
                        j
                    )
                    num_regions += 1
        self.regions._check_map()

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr

    @property
    def abstract(self):
        return self._abstract

    @property
    def ndim(self):
        """The dimensions of this manifold (Not the embedding space dimensions!)"""
        return self._abstract.ndim

    @property
    def esd(self):
        """embedding space dimensions."""
        return get_embedding_space_dim()

    @property
    def m(self):
        return self.esd

    @property
    def n(self):
        return self.ndim

    @property
    def ct(self):
        if self._ct is None:
            self._ct = MsePyManifoldsCoordinateTransformation(self)
        return self._ct

    @property
    def regions(self):
        """"""
        assert self._regions is not None, f"regions of {self} is not configured, config it through `msepy.config`"
        return self._regions

    @property
    def visualize(self):
        if self._visualize is None:
            self._visualize = MsePyManifoldVisualize(self)
        return self._visualize
