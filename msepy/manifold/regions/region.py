# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 4/3/2023 5:46 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen


class MsePyManifoldRegion(Frozen):
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

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Region#{self._i} of " + super_repr


if __name__ == '__main__':
    # mpiexec -n 4 python 

    pass
