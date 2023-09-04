# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2MeshElementsCoordinateTransformation(Frozen):
    """"""

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._freeze()

    def check_fc_range(self, fc_range):
        """"""
        if fc_range is None:
            fc_range = self._elements._fundamental_cells
        else:
            if fc_range.__class__.__name__ in ('float', 'int', 'int32', 'int64'):
                fc_range = [fc_range, ]
            else:
                pass

            for i in fc_range:
                assert i in self._elements, f"element #{i} is out of range!"

        return fc_range

    def mapping(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        mp = dict()
        for e in fc_range:
            mp[e] = self._elements[e].ct.mapping(xi, et)
        return mp

    def Jacobian_matrix(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        JM_cache = dict()
        JM = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if cache_index in JM_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                JM_cache[cache_index] = self._elements[e].ct.Jacobian_matrix(xi, et)

            JM[e] = JM_cache[cache_index]

        return JM

    def Jacobian(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        J_cache = dict()
        J = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if cache_index in J_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                J_cache[cache_index] = self._elements[e].ct.Jacobian(xi, et)

            J[e] = J_cache[cache_index]

        return J

    def metric(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        m_cache = dict()
        m = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if cache_index in m_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                m_cache[cache_index] = self._elements[e].ct.metric(xi, et)

            m[e] = m_cache[cache_index]

        return m

    def inverse_Jacobian_matrix(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        iJM_cache = dict()
        iJM = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if cache_index in iJM_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                iJM_cache[cache_index] = self._elements[e].ct.inverse_Jacobian_matrix(xi, et)

            iJM[e] = iJM_cache[cache_index]

        return iJM

    def inverse_Jacobian(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        iJ_cache = dict()
        iJ = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if cache_index in iJ_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                iJ_cache[cache_index] = self._elements[e].ct.inverse_Jacobian(xi, et)

            iJ[e] = iJ_cache[cache_index]

        return iJ

    def metric_matrix(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        mm_cache = dict()
        mm = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if cache_index in mm_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                mm_cache[cache_index] = self._elements[e].ct.metric_matrix(xi, et)

            mm[e] = mm_cache[cache_index]

        return mm

    def inverse_metric_matrix(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        imm_cache = dict()
        imm = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if cache_index in imm_cache:
                pass
            else:  # compute the data for elements of type: ``cache_index``.
                imm_cache[cache_index] = self._elements[e].ct.inverse_metric_matrix(xi, et)

            imm[e] = imm_cache[cache_index]

        return imm
