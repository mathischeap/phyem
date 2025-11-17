# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


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
        else:
            if element_range.__class__.__name__ in ('float', 'int', 'int32', 'int64'):
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

            JM[e] = JM_cache[cache_index]

        return JM

    def Jacobian(self, *xi_et_sg, element_range=None):
        """"""
        element_range = self._check_element_range(element_range)
        J_cache = dict()
        J = dict()
        for e in element_range:
            cache_index = self._e2c[e]
            if cache_index in J_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                J_cache[cache_index] = self._elements[e].ct.Jacobian(*xi_et_sg)

            J[e] = J_cache[cache_index]

        return J

    def metric(self, *xi_et_sg, element_range=None):
        """"""
        element_range = self._check_element_range(element_range)
        m_cache = dict()
        m = dict()
        for e in element_range:
            cache_index = self._e2c[e]
            if cache_index in m_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                m_cache[cache_index] = self._elements[e].ct.metric(*xi_et_sg)

            m[e] = m_cache[cache_index]

        return m

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

            iJM[e] = iJM_cache[cache_index]

        return iJM

    def inverse_Jacobian(self, *xi_et_sg, element_range=None):
        """"""
        element_range = self._check_element_range(element_range)
        iJ_cache = dict()
        iJ = dict()
        for e in element_range:
            cache_index = self._e2c[e]
            if cache_index in iJ_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                iJ_cache[cache_index] = self._elements[e].ct.inverse_Jacobian(*xi_et_sg)

            iJ[e] = iJ_cache[cache_index]

        return iJ

    def metric_matrix(self, *xi_et_sg, element_range=None):
        """"""
        element_range = self._check_element_range(element_range)
        mm_cache = dict()
        mm = dict()
        for e in element_range:
            cache_index = self._e2c[e]
            if cache_index in mm_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                mm_cache[cache_index] = self._elements[e].ct.metric_matrix(*xi_et_sg)

            mm[e] = mm_cache[cache_index]

        return mm

    def inverse_metric_matrix(self, *xi_et_sg, element_range=None):
        """"""
        element_range = self._check_element_range(element_range)
        imm_cache = dict()
        imm = dict()
        for e in element_range:
            cache_index = self._e2c[e]
            if cache_index in imm_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                imm_cache[cache_index] = self._elements[e].ct.inverse_metric_matrix(*xi_et_sg)

            imm[e] = imm_cache[cache_index]

        return imm
