# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer
_global_JM_cache = dict()
_global_J_cache = dict()
_global_m_cache = dict()
_global_iJM_cache = dict()
_global_iJ_cache = dict()
_global_mm_cache = dict()
_global_imm_cache = dict()


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
        JM = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if isinstance(cache_index, int):  # unique fc, no cache
                JM[e] = self._elements[e].ct.Jacobian_matrix(xi, et)
            else:

                cached, jm = ndarray_key_comparer(_global_JM_cache, [xi, et], check_str=cache_index)
                if cached:
                    pass
                else:
                    jm = self._elements[e].ct.Jacobian_matrix(xi, et)
                    add_to_ndarray_cache(_global_JM_cache, [xi, et], jm, check_str=cache_index, maximum=16)

                JM[e] = jm

        return JM

    def Jacobian(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        J = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if isinstance(cache_index, int):  # unique fc, no cache
                J[e] = self._elements[e].ct.Jacobian(xi, et)
            else:
                cached, j = ndarray_key_comparer(_global_J_cache, [xi, et], check_str=cache_index)
                if cached:
                    pass
                else:
                    j = self._elements[e].ct.Jacobian(xi, et)
                    add_to_ndarray_cache(_global_J_cache, [xi, et], j, check_str=cache_index, maximum=16)

                J[e] = j

        return J

    def metric(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        m = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if isinstance(cache_index, int):  # unique fc, no cache
                m[e] = self._elements[e].ct.metric(xi, et)
            else:

                cached, mt = ndarray_key_comparer(_global_m_cache, [xi, et], check_str=cache_index)

                if cached:
                    pass
                else:  # compute the data for elements of type: ``cache_index``.
                    mt = self._elements[e].ct.metric(xi, et)
                    add_to_ndarray_cache(_global_m_cache, [xi, et], mt, check_str=cache_index, maximum=16)

                m[e] = mt

        return m

    def inverse_Jacobian_matrix(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        iJM = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if isinstance(cache_index, int):  # unique fc, no cache
                iJM[e] = self._elements[e].ct.inverse_Jacobian_matrix(xi, et)
            else:

                cached, ijm = ndarray_key_comparer(_global_iJM_cache, [xi, et], check_str=cache_index)
                if cached:
                    pass
                else:
                    ijm = self._elements[e].ct.inverse_Jacobian_matrix(xi, et)
                    add_to_ndarray_cache(_global_iJM_cache, [xi, et], ijm, check_str=cache_index, maximum=16)

                iJM[e] = ijm

        return iJM

    def inverse_Jacobian(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        iJ = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if isinstance(cache_index, int):  # unique fc, no cache
                iJ[e] = self._elements[e].ct.inverse_Jacobian(xi, et)
            else:

                cached, ij = ndarray_key_comparer(_global_iJ_cache, [xi, et], check_str=cache_index)

                if cached:
                    pass
                else:  # compute the data for elements of type: ``cache_index``.
                    ij = self._elements[e].ct.inverse_Jacobian(xi, et)
                    add_to_ndarray_cache(_global_iJ_cache, [xi, et], ij, check_str=cache_index, maximum=16)

                iJ[e] = ij

        return iJ

    def metric_matrix(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)
        mm = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature

            if isinstance(cache_index, int):  # unique fc, no cache
                mm[e] = self._elements[e].ct.metric_matrix(xi, et)
            else:

                cached, mmt = ndarray_key_comparer(_global_mm_cache, [xi, et], check_str=cache_index)

                if cached:
                    pass
                else:  # compute the data for elements of type: ``cache_index``.
                    mmt = self._elements[e].ct.metric_matrix(xi, et)
                    add_to_ndarray_cache(_global_mm_cache, [xi, et], mmt, check_str=cache_index, maximum=16)

                mm[e] = mmt

        return mm

    def inverse_metric_matrix(self, xi, et, fc_range=None):
        """"""
        fc_range = self.check_fc_range(fc_range)

        imm = dict()
        for e in fc_range:
            cache_index = self._elements[e].metric_signature
            if isinstance(cache_index, int):  # unique fc, no cache
                imm[e] = self._elements[e].ct.inverse_metric_matrix(xi, et)
            else:
                # cached, i_mmt = ndarray_key_comparer(
                #     _global_imm_cache, [xi, et], check_str=cache_index
                # )
                #
                # if cached:
                #     pass
                # else:  # compute the data for elements of type: ``cache_index``.
                i_mmt = self._elements[e].ct.inverse_metric_matrix(xi, et)
                    # add_to_ndarray_cache(
                    #     _global_imm_cache, [xi, et], i_mmt, check_str=cache_index, maximum=16
                    # )

                imm[e] = i_mmt

        return imm
