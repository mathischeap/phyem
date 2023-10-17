# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class Element(Frozen):
    """"""

    @property
    def type(self):
        raise NotImplementedError()

    @property
    def orthogonal(self):
        raise NotImplementedError()

    @property
    def metric_signature(self):
        raise NotImplementedError()

    @property
    def ct(self):
        raise NotImplementedError()

    @property
    def area(self):
        raise NotImplementedError()

    def _plot_lines(self, density):
        raise NotImplementedError()


_global_JM_cache_CT = {}
_global_J_cache_CT = {}
_global_m_cache_CT = {}
_global_iJM_cache_CT = {}
_global_iJ_cache_CT = {}
_global_mm_cache_CT = {}
_global_iMM_cache_CT = {}
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


class CoordinateTransformation:
    """"""
    def __init__(self, metric_signature):
        self._metric_signature = metric_signature

    def mapping(self, xi, et):
        """"""
        raise NotImplementedError

    def Jacobian_matrix(self, xi, et):
        """ r, s be in [-1, 1]. """
        cached, JM = ndarray_key_comparer(
            _global_JM_cache_CT,
            [xi, et],
            check_str=self._metric_signature
        )
        if cached:
            pass
        else:
            JM = self.___Jacobian_matrix___(xi, et)
            add_to_ndarray_cache(
                _global_JM_cache_CT,
                [xi, et],
                JM,
                check_str=self._metric_signature,
                maximum=16,
            )
        return JM

    def ___Jacobian_matrix___(self, xi, et):
        raise NotImplementedError

    def Jacobian(self, xi, et):
        """Determinant of the Jacobian matrix."""
        cached, data = ndarray_key_comparer(
            _global_J_cache_CT,
            [xi, et],
            check_str=self._metric_signature
        )
        if cached:
            pass
        else:
            J = self.Jacobian_matrix(xi, et)
            data = J[0][0]*J[1][1] - J[0][1]*J[1][0]
            add_to_ndarray_cache(
                _global_J_cache_CT, [xi, et], data, check_str=self._metric_signature,
                maximum=16,
            )
        return data

    def metric(self, *evaluationPoints):
        """
        The metric ``g:= det(G):=(det(J))**2``. Since our Jacobian and inverse of Jacobian are both square,
        we know that the metric ``g`` is equal to square of ``det(J)``. ``g = (det(J))**2`` is due to the
        fact that the Jacobian matrix is square. The definition of ``g`` usually is given
        as ``g:= det(G)`` where ``G`` is the metric matrix, or metric tensor.
        """
        cached, m = ndarray_key_comparer(
            _global_m_cache_CT,
            evaluationPoints,
            check_str=self._metric_signature
        )
        if cached:
            pass
        else:
            detJ = self.Jacobian(*evaluationPoints)
            m = detJ ** 2
            add_to_ndarray_cache(
                _global_m_cache_CT, evaluationPoints, m, check_str=self._metric_signature,
                maximum=16,
            )
        return m

    def inverse_Jacobian_matrix(self, *evaluationPoints):
        """The inverse Jacobian matrix. """
        cached, iJM = ndarray_key_comparer(
            _global_iJM_cache_CT, evaluationPoints, check_str=self._metric_signature
        )
        if cached:
            pass
        else:
            J = self.Jacobian_matrix(*evaluationPoints)
            Jacobian = J[0][0]*J[1][1] - J[0][1]*J[1][0]
            reciprocalJacobian = 1 / Jacobian
            del Jacobian
            iJ00 = + reciprocalJacobian * J[1][1]
            iJ01 = - reciprocalJacobian * J[0][1]
            iJ10 = - reciprocalJacobian * J[1][0]
            iJ11 = + reciprocalJacobian * J[0][0]
            iJM = [[iJ00, iJ01],
                   [iJ10, iJ11]]
            add_to_ndarray_cache(
                _global_iJM_cache_CT, evaluationPoints, iJM, check_str=self._metric_signature,
                maximum=16,
            )
        return iJM

    def inverse_Jacobian(self, *evaluationPoints):
        """Determinant of the inverse Jacobian matrix. """
        cached, J = ndarray_key_comparer(
            _global_iJ_cache_CT, evaluationPoints, check_str=self._metric_signature
        )
        if cached:
            pass
        else:
            iJ = self.inverse_Jacobian_matrix(*evaluationPoints)
            J = iJ[0][0]*iJ[1][1] - iJ[0][1]*iJ[1][0]
            add_to_ndarray_cache(
                _global_iJ_cache_CT, evaluationPoints, J, check_str=self._metric_signature,
                maximum=16,
            )
        return J

    def metric_matrix(self, *evaluationPoints):
        """
        Also called metric tensor. Let J be the Jacobian matrix. The ``metricMatrix`` is
        denoted by G, G := J^T.dot(J). And the metric is ``g := (det(J))**2 or g := det(G).``
        Which means for a square Jacobian matrix, the metric turns out to be the square of the
        determinant of the Jacobian matrix.

        The entries of G are normally denoted as g_{i,j}.
        """
        cached, G = ndarray_key_comparer(
            _global_mm_cache_CT, evaluationPoints, check_str=self._metric_signature
        )
        if cached:
            pass
        else:
            J = self.Jacobian_matrix(*evaluationPoints)
            G = [[None for _ in range(2)] for __ in range(2)]
            for i in range(2):
                for j in range(i, 2):
                    # noinspection PyTypeChecker
                    G[i][j] = J[0][i] * J[0][j]
                    for l_ in range(1, 2):
                        G[i][j] += J[l_][i] * J[l_][j]
                    if i != j:
                        G[j][i] = G[i][j]
            add_to_ndarray_cache(
                _global_mm_cache_CT, evaluationPoints, G, check_str=self._metric_signature,
                maximum=16,
            )
        return G

    def inverse_metric_matrix(self, *evaluationPoints):
        """
        The ``inverseMetricMatrix`` is the metric matrix of the inverse Jacobian matrix
        or the metric of the inverse mapping. It is usually denoted as G^{-1}.

        The entries of G^{-1} are normally denoted as g^{i,j}.
        """
        cached, iG = ndarray_key_comparer(
            _global_iMM_cache_CT, evaluationPoints, check_str=self._metric_signature
        )
        if cached:
            pass
        else:
            iJ = self.inverse_Jacobian_matrix(*evaluationPoints)
            iG = [[None for _ in range(2)] for __ in range(2)]
            for i in range(2):
                for j in range(i, 2):
                    # noinspection PyTypeChecker
                    iG[i][j] = iJ[i][0] * iJ[j][0]
                    for l_ in range(1, 2):
                        iG[i][j] += iJ[i][l_] * iJ[j][l_]
                    if i != j:
                        iG[j][i] = iG[i][j]
            add_to_ndarray_cache(
                _global_iMM_cache_CT, evaluationPoints, iG, check_str=self._metric_signature,
                maximum=16,
            )
        return iG
