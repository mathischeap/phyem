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

from msepy.manifold.regions.region import MsePyManifoldRegion
from msepy.manifold.regions.rct import MsePyRegionCoordinateTransformation


def config(mf, arg, **kwargs):
    """"""
    assert mf.__class__ is MsePyManifold, f"I can only config MseManifold instance. Now I get {mf}."
    assert mf._regions is None, f"manifold {mf} already have region configurations. Change it may lead to error."

    mf._regions = MseManifoldRegions(mf)  # initialize the regions.

    if isinstance(arg, str):  # use predefined mappings
        predefined_path = '.'.join(str(MsePyManifold).split(' ')[1][1:-2].split('.')[:-2]) + \
                          '.predefined.' + arg
        _module = import_module(predefined_path)
        region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict = getattr(_module, arg)(mf, **kwargs)

        mf._parse_regions_from_region_map(
            region_map,
            mapping_dict,
            Jacobian_matrix_dict,
            mtype_dict
        )

        map_type = 0
        assert mf.regions.map is not None, f"predefined manifold only config manifold with region map."

    else:
        raise NotImplementedError()

    assert mf.regions._regions != dict(), f"we need to set regions for the manifold by this `config` function."
    mf.regions._check_map(map_type)


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
            region_map,
            mapping_dict,
            Jacobian_matrix_dict,
            mtype_dict
    ):
        assert self.regions._regions == dict(), f"Change regions will be dangerous!"
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

            rct = MsePyRegionCoordinateTransformation(mapping, Jacobian_matrix, mtype)
            region = MsePyManifoldRegion(self, i, rct)
            self.regions._regions[i] = region

        self.regions._map = region_map  # ***
        assert self.abstract._is_periodic is self.regions._is_periodic(), f"Periodicity does not match."

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
