# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2MeshElementsLevelTrianglesCT(Frozen):
    """Not a level ct. Call me from ``level.elements``."""

    def __init__(self, level_triangles):
        """"""
        self._triangles = level_triangles
        self._freeze()

    def _check_triangle_range(self, triangle_range):
        """"""
        if triangle_range is None:
            triangle_range = self._triangles._triangle_dict
        else:
            if triangle_range.__class__.__name__ in ('float', 'int', 'int32', 'int64'):
                triangle_range = [triangle_range, ]
            else:
                pass

            for i in triangle_range:
                assert i in self._triangles, f"element #{i} is out of range!"

        return triangle_range

    def mapping(self, xi, et, triangle_range=None):
        """"""
        triangle_range = self._check_triangle_range(triangle_range)
        mp = dict()
        for e in triangle_range:
            mp[e] = self._triangles[e].ct.mapping(xi, et)
        return mp

    def Jacobian_matrix(self, xi, et, triangle_range=None):
        """"""
        triangle_range = self._check_triangle_range(triangle_range)
        JM_cache = dict()
        JM = dict()
        for e in triangle_range:
            cache_index = self._triangles[e].metric_signature
            if cache_index in JM_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                JM_cache[cache_index] = self._triangles[e].ct.Jacobian_matrix(xi, et)

            JM[e] = JM_cache[cache_index]

        return JM

    def Jacobian(self, xi, et, triangle_range=None):
        """"""
        triangle_range = self._check_triangle_range(triangle_range)
        J_cache = dict()
        J = dict()
        for e in triangle_range:
            cache_index = self._triangles[e].metric_signature
            if cache_index in J_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                J_cache[cache_index] = self._triangles[e].ct.Jacobian(xi, et)

            J[e] = J_cache[cache_index]

        return J

    def metric(self, xi, et, triangle_range=None):
        """"""
        triangle_range = self._check_triangle_range(triangle_range)
        m_cache = dict()
        m = dict()
        for e in triangle_range:
            cache_index = self._triangles[e].metric_signature
            if cache_index in m_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                m_cache[cache_index] = self._triangles[e].ct.metric(xi, et)

            m[e] = m_cache[cache_index]

        return m

    def inverse_Jacobian_matrix(self, xi, et, triangle_range=None):
        """"""
        triangle_range = self._check_triangle_range(triangle_range)
        iJM_cache = dict()
        iJM = dict()
        for e in triangle_range:
            cache_index = self._triangles[e].metric_signature
            if cache_index in iJM_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                iJM_cache[cache_index] = self._triangles[e].ct.inverse_Jacobian_matrix(xi, et)

            iJM[e] = iJM_cache[cache_index]

        return iJM

    def inverse_Jacobian(self, xi, et, triangle_range=None):
        """"""
        triangle_range = self._check_triangle_range(triangle_range)
        iJ_cache = dict()
        iJ = dict()
        for e in triangle_range:
            cache_index = self._triangles[e].metric_signature
            if cache_index in iJ_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                iJ_cache[cache_index] = self._triangles[e].ct.inverse_Jacobian(xi, et)

            iJ[e] = iJ_cache[cache_index]

        return iJ

    def metric_matrix(self, xi, et, triangle_range=None):
        """"""
        triangle_range = self._check_triangle_range(triangle_range)
        mm_cache = dict()
        mm = dict()
        for e in triangle_range:
            cache_index = self._triangles[e].metric_signature
            if cache_index in mm_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                mm_cache[cache_index] = self._triangles[e].ct.metric_matrix(xi, et)

            mm[e] = mm_cache[cache_index]

        return mm

    def inverse_metric_matrix(self, xi, et, triangle_range=None):
        """"""
        triangle_range = self._check_triangle_range(triangle_range)
        imm_cache = dict()
        imm = dict()
        for e in triangle_range:
            cache_index = self._triangles[e].metric_signature
            if cache_index in imm_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                imm_cache[cache_index] = self._triangles[e].ct.inverse_metric_matrix(xi, et)

            imm[e] = imm_cache[cache_index]

        return imm
