# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class MsePyManifoldStandardRegion(Frozen):
    """"""

    def __init__(
            self,
            regions, i, rct
    ):
        """
        Parameters
        ----------
        regions :
            The group of regions this region belong to.
        i :
            This is the ith region in the regions of a manifolds
        rct:
            The ct that maps the reference region into this region.
        """
        self._regions = regions
        self._i = i
        self._ct = rct
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Region#{self._i} of {self._regions._mf}" + super_repr
