# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen

from phyem.msehtt.static.space.integrate_matrix_over_sub_geometry.Lambda.main import MseHttSpace_iMSG_Lambda


class MseHttSpace_IntMatOverSubGeo(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        indicator = self._space.indicator
        if indicator == 'Lambda':
            self._particular_space_integrator = MseHttSpace_iMSG_Lambda(space)
        else:
            raise NotImplementedError(f"IntOverSubGeo not coded for {indicator}-space.")
        self._freeze()

    def __call__(self, degree, element_index, *geo_coo, **kwargs):
        r""""""
        return self._particular_space_integrator(degree, element_index, *geo_coo, **kwargs)
