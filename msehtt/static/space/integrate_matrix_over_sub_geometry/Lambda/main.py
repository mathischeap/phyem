# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen

from phyem.msehtt.static.space.integrate_matrix_over_sub_geometry.Lambda.ImSG_m2n0k0 import ___iMsg_m2n2k0___
from phyem.msehtt.static.space.integrate_matrix_over_sub_geometry.Lambda.ImSG_m2n2k1 import ___iMsg_m2n2k1_inner___
from phyem.msehtt.static.space.integrate_matrix_over_sub_geometry.Lambda.ImSG_m2n2k1 import ___iMsg_m2n2k1_outer___
from phyem.msehtt.static.space.integrate_matrix_over_sub_geometry.Lambda.ImSG_m2n2k2 import ___iMsg_m2n2k2___


class MseHttSpace_iMSG_Lambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        self._orientation = self._space.orientation
        self._indicator = f"m{m}n{n}k{k}"
        self._freeze()

    def __call__(self, degree, element_index, *geo_coo, **kwargs):
        r""""""
        elements = self._space.tpm.composition
        element = elements[element_index]

        if self._indicator == 'm2n2k0':
            return ___iMsg_m2n2k0___(degree, element, *geo_coo)
        elif self._indicator == 'm2n2k1':
            if self._orientation == 'inner':
                return ___iMsg_m2n2k1_inner___(degree, element, *geo_coo, **kwargs)
            elif self._orientation == 'outer':
                return ___iMsg_m2n2k1_outer___(degree, element, *geo_coo, **kwargs)
            else:
                raise Exception()
        elif self._indicator == 'm2n2k2':
            return ___iMsg_m2n2k2___(degree, element, *geo_coo)
        else:
            raise NotImplementedError()
