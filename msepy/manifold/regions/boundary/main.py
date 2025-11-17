# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msepy.manifold.regions.boundary.ct import MsePyBoundaryRegionCoordinateTransformation


class MsePyManifoldBoundaryRegion(Frozen):
    """"""

    def __init__(
            self,
            i,
            boundary_regions,
            base_region,
            side_index,
    ):
        """"""
        self._i = i  # the ith boundary region
        self._regions = boundary_regions
        self._base_region = base_region
        self._side_index = side_index
        self._make_rct()
        self._freeze()

    def _make_rct(self):
        """"""
        self._ct = MsePyBoundaryRegionCoordinateTransformation(self)

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<BoundaryRegion#{self._i} of {self._regions._mf}" + super_repr

    @property
    def m(self):
        """esd"""
        return self._regions.m

    @property
    def n(self):
        """ndim"""
        return self._regions.n
