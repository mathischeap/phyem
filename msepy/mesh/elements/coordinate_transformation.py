# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:20 PM on 7/17/2023
"""
from tools.frozen import Frozen


class MsePyMeshElementsCooTrans(Frozen):
    """Compute ct data for particular elements. Always return a dict whose keys are the elements."""

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._mesh = elements._mesh
        self._e2c = elements._index_mapping._e2c
        self._freeze()

    def _check_element_range(self, element_range):
        """"""

        if element_range is None:
            element_range = range(self._mesh.elements._num)
        elif element_range.__class__.__name__ in ('float', 'int', 'int32', 'int64'):
            element_range = [element_range, ]
        else:
            pass
        for i in element_range:
            assert i in self._elements, f"element #{i} is out of range!"

        return element_range

    def mapping(self, *xi_et_sg, element_range=None):
        """"""
        element_range = self._check_element_range(element_range)
        mp = dict()
        for e in element_range:
            mp[e] = self._elements[e].ct.mapping(*xi_et_sg)
        return mp

    def Jacobian_matrix(self, *xi_et_sg, element_range=None):
        """"""
        element_range = self._check_element_range(element_range)
        JM_cache = dict()
        JM = dict()
        for e in element_range:
            cache_index = self._e2c[e]
            if cache_index in JM_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                JM_cache[cache_index] = self._elements[e].ct.Jacobian_matrix(*xi_et_sg)

            JM_e = JM_cache[cache_index]
            JM[e] = JM_e
        return JM

    def inverse_Jacobian_matrix(self, *xi_et_sg, element_range=None):
        """"""
        element_range = self._check_element_range(element_range)
        iJM_cache = dict()
        iJM = dict()
        for e in element_range:
            cache_index = self._e2c[e]
            if cache_index in iJM_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                iJM_cache[cache_index] = self._elements[e].ct.inverse_Jacobian_matrix(*xi_et_sg)

            iJM_e = iJM_cache[cache_index]
            iJM[e] = iJM_e
        return iJM
