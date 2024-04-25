# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from tools.frozen import Frozen


class MseHttGreatMeshBaseElement(Frozen):
    """"""

    def __init__(self):
        """"""
        self._index = None
        self._map = None
        self._parameters = None
        self._ct = None
        self._m = None   # the dimensions of the space
        self._n = None   # the dimensions of the element
        self._faces = None
        self._freeze()

    @property
    def index(self):
        """The index of this element. Must be unique all over the great mesh."""
        return self._index

    @property
    def m(self):
        """the dimensions of the space"""
        return self._m

    @property
    def n(self):
        """the dimensions of the element"""
        return self._n

    @property
    def etype(self):
        """Return the indicator of the element type."""
        return self._etype()

    @classmethod
    def _etype(cls):
        raise NotImplementedError()

    @property
    def parameters(self):
        return self._parameters

    @property
    def metric_signature(self):
        raise NotImplementedError()

    @property
    def map_(self):
        """The numbering of the nodes of the element.

        For example, for a rectangle element, element.map_ = [0, 100, 54 33]. This means the four nodes of
        this element are labeled 0, 100, 54 33 respectively.
        """
        return self._map

    @property
    def ct(self):
        return self._ct

    @property
    def _generate_outline_data(self, ddf=1):
        raise NotImplementedError()

    @classmethod
    def face_setting(cls):
        """The face setting; show nodes of each face of the element and the positive direction.

        A face can be 2-d (face of 3d element) or 1-d (face of 2d element).
        """
        raise NotImplementedError()


# ============ ELEMENT CT =====================================================================================
___cache_msehtt_JM___ = {}
___cache_msehtt_Jacobian___ = {}
___cache_msehtt_inverseJacobian___ = {}
___cache_msehtt_iJM___ = {}
___cache_msehtt_metric___ = {}
___cache_msehtt_mm___ = {}
___cache_msehtt_imm___ = {}


def ___clean_cache_msehtt_element_ct___():
    """"""
    keys = list(___cache_msehtt_JM___.keys())
    for key in keys:
        del ___cache_msehtt_JM___[key]
    keys = list(___cache_msehtt_Jacobian___.keys())
    for key in keys:
        del ___cache_msehtt_Jacobian___[key]
    keys = list(___cache_msehtt_inverseJacobian___.keys())
    for key in keys:
        del ___cache_msehtt_inverseJacobian___[key]
    keys = list(___cache_msehtt_iJM___.keys())
    for key in keys:
        del ___cache_msehtt_iJM___[key]
    keys = list(___cache_msehtt_metric___.keys())
    for key in keys:
        del ___cache_msehtt_metric___[key]
    keys = list(___cache_msehtt_mm___.keys())
    for key in keys:
        del ___cache_msehtt_mm___[key]
    keys = list(___cache_msehtt_imm___.keys())
    for key in keys:
        del ___cache_msehtt_imm___[key]


from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


class MseHttGreatMeshBaseElementCooTrans(Frozen):
    """"""
    def __init__(self, element, metric_signature):
        self._element = element
        self._metric_signature = metric_signature
        self._freeze()

    def __repr__(self):
        """"""
        return f"<CT of {self._element.__repr__()}>"

    def mapping(self, *xi_et_sg):
        raise NotImplementedError()

    def ___Jacobian_matrix___(self, *xi_et_sg):
        raise NotImplementedError()

    def Jacobian_matrix(self, *xi_et_sg):
        if isinstance(self._metric_signature, int):  # unique element, no cache
            return self.___Jacobian_matrix___(*xi_et_sg)
        else:
            cached, data = ndarray_key_comparer(___cache_msehtt_JM___, xi_et_sg, check_str=self._metric_signature)
            if cached:
                pass
            else:
                data = self.___Jacobian_matrix___(*xi_et_sg)
                add_to_ndarray_cache(___cache_msehtt_JM___, xi_et_sg, data, check_str=self._metric_signature)
            return data

    def inverse_Jacobian_matrix(self, *xi_et_sg):
        """"""
        if isinstance(self._metric_signature, int):  # unique element, no cache
            return self.___inverse_Jacobian_matrix___(*xi_et_sg)
        else:
            cached, data = ndarray_key_comparer(___cache_msehtt_iJM___, xi_et_sg, check_str=self._metric_signature)
            if cached:
                pass
            else:
                data = self.___inverse_Jacobian_matrix___(*xi_et_sg)
                add_to_ndarray_cache(___cache_msehtt_iJM___, xi_et_sg, data, check_str=self._metric_signature)
            return data

    def ___inverse_Jacobian_matrix___(self, *xi_et_sg):
        """"""
        jm = self.Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m, self._element.n

        if m == n == 1:

            iJM00 = 1 / jm[0][0]
            ijm = [[iJM00, ], ]

        elif m == n == 2:

            reciprocalJacobian = 1 / (jm[0][0] * jm[1][1] - jm[0][1] * jm[1][0])
            iJ00 = + reciprocalJacobian * jm[1][1]
            iJ01 = - reciprocalJacobian * jm[0][1]
            iJ10 = - reciprocalJacobian * jm[1][0]
            iJ11 = + reciprocalJacobian * jm[0][0]
            ijm = \
                [
                    [iJ00, iJ01],
                    [iJ10, iJ11]
                ]

        elif m == n == 3:

            Jacobian = \
                + jm[0][0] * jm[1][1] * jm[2][2] + jm[0][1] * jm[1][2] * jm[2][0] \
                + jm[0][2] * jm[1][0] * jm[2][1] - jm[0][0] * jm[1][2] * jm[2][1] \
                - jm[0][1] * jm[1][0] * jm[2][2] - jm[0][2] * jm[1][1] * jm[2][0]

            reciprocalJacobian = 1 / Jacobian

            iJ00 = reciprocalJacobian * (jm[1][1] * jm[2][2] - jm[1][2] * jm[2][1])
            iJ01 = reciprocalJacobian * (jm[2][1] * jm[0][2] - jm[2][2] * jm[0][1])
            iJ02 = reciprocalJacobian * (jm[0][1] * jm[1][2] - jm[0][2] * jm[1][1])
            iJ10 = reciprocalJacobian * (jm[1][2] * jm[2][0] - jm[1][0] * jm[2][2])
            iJ11 = reciprocalJacobian * (jm[2][2] * jm[0][0] - jm[2][0] * jm[0][2])
            iJ12 = reciprocalJacobian * (jm[0][2] * jm[1][0] - jm[0][0] * jm[1][2])
            iJ20 = reciprocalJacobian * (jm[1][0] * jm[2][1] - jm[1][1] * jm[2][0])
            iJ21 = reciprocalJacobian * (jm[2][0] * jm[0][1] - jm[2][1] * jm[0][0])
            iJ22 = reciprocalJacobian * (jm[0][0] * jm[1][1] - jm[0][1] * jm[1][0])

            ijm = [
                [iJ00, iJ01, iJ02],
                [iJ10, iJ11, iJ12],
                [iJ20, iJ21, iJ22]
            ]

        else:
            raise NotImplementedError()

        return ijm

    def Jacobian(self, *xi_et_sg):
        """the Determinant of the Jacobian matrix. When Jacobian matrix is square, Jacobian = sqrt(g)."""
        if isinstance(self._metric_signature, int):  # unique element, no cache
            return self.___Jacobian___(*xi_et_sg)
        else:
            cached, data = ndarray_key_comparer(___cache_msehtt_Jacobian___, xi_et_sg, check_str=self._metric_signature)
            if cached:
                pass
            else:
                data = self.___Jacobian___(*xi_et_sg)
                add_to_ndarray_cache(___cache_msehtt_Jacobian___, xi_et_sg, data, check_str=self._metric_signature)
            return data

    def ___Jacobian___(self, *xi_et_sg):
        """"""
        jm = self.Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m, self._element.n

        if m == n == 1:

            Jacobian = jm[0][0]

        elif m == n == 2:

            Jacobian = jm[0][0] * jm[1][1] - jm[0][1] * jm[1][0]

        elif m == n == 3:

            Jacobian = \
                + jm[0][0] * jm[1][1] * jm[2][2] + jm[0][1] * jm[1][2] * jm[2][0] \
                + jm[0][2] * jm[1][0] * jm[2][1] - jm[0][0] * jm[1][2] * jm[2][1] \
                - jm[0][1] * jm[1][0] * jm[2][2] - jm[0][2] * jm[1][1] * jm[2][0]

        else:
            raise NotImplementedError()

        return Jacobian

    def metric(self, *xi_et_sg):
        """ For square Jacobian matrix,
        the metric ``g:= det(G):=(det(J))**2``, where ``G`` is the metric matrix, or metric tensor.
        """
        if isinstance(self._metric_signature, int):  # unique element, no cache
            return self.___metric___(*xi_et_sg)
        else:
            cached, data = ndarray_key_comparer(___cache_msehtt_metric___, xi_et_sg, check_str=self._metric_signature)
            if cached:
                pass
            else:
                data = self.___metric___(*xi_et_sg)
                add_to_ndarray_cache(___cache_msehtt_metric___, xi_et_sg, data, check_str=self._metric_signature)
            return data

    def ___metric___(self, *xi_et_sg):
        """"""
        m, n = self._element.m, self._element.n
        if m == n:
            return self.Jacobian(*xi_et_sg) ** 2
        else:
            raise NotImplementedError()

    def inverse_Jacobian(self, *xi_et_sg):
        """the Determinant of the inverse Jacobian matrix."""
        if isinstance(self._metric_signature, int):  # unique element, no cache
            return self.___inverse_Jacobian___(*xi_et_sg)
        else:
            cached, data = ndarray_key_comparer(
                ___cache_msehtt_inverseJacobian___, xi_et_sg, check_str=self._metric_signature
            )
            if cached:
                pass
            else:
                data = self.___inverse_Jacobian___(*xi_et_sg)
                add_to_ndarray_cache(
                    ___cache_msehtt_inverseJacobian___, xi_et_sg, data, check_str=self._metric_signature
                )
            return data

    def ___inverse_Jacobian___(self, *xi_et_sg):
        """"""
        ijm = self.inverse_Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m, self._element.n

        if m == n == 1:

            inverse_Jacobian = ijm[0][0]

        elif m == n == 2:

            inverse_Jacobian = ijm[0][0] * ijm[1][1] - ijm[0][1] * ijm[1][0]

        elif m == n == 3:

            inverse_Jacobian = \
                + ijm[0][0] * ijm[1][1] * ijm[2][2] + ijm[0][1] * ijm[1][2] * ijm[2][0] \
                + ijm[0][2] * ijm[1][0] * ijm[2][1] - ijm[0][0] * ijm[1][2] * ijm[2][1] \
                - ijm[0][1] * ijm[1][0] * ijm[2][2] - ijm[0][2] * ijm[1][1] * ijm[2][0]

        else:
            raise NotImplementedError()

        return inverse_Jacobian

    def metric_matrix(self, *xi_et_sg):
        """"""
        if isinstance(self._metric_signature, int):  # unique element, no cache
            return self.___metric_matrix___(*xi_et_sg)
        else:
            cached, data = ndarray_key_comparer(___cache_msehtt_mm___, xi_et_sg, check_str=self._metric_signature)
            if cached:
                pass
            else:
                data = self.___metric_matrix___(*xi_et_sg)
                add_to_ndarray_cache(___cache_msehtt_mm___, xi_et_sg, data, check_str=self._metric_signature)
            return data

    def ___metric_matrix___(self, *xi_et_sg):
        """"""
        jm = self.Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m, self._element.n
        G = [[None for _ in range(n)] for __ in range(n)]
        for i in range(n):
            for j in range(i, n):
                # noinspection PyTypeChecker
                G[i][j] = jm[0][i] * jm[0][j]
                for L in range(1, m):
                    G[i][j] += jm[L][i] * jm[L][j]
                if i != j:
                    G[j][i] = G[i][j]
        return G

    def inverse_metric_matrix(self, *xi_et_sg):
        """"""
        if isinstance(self._metric_signature, int):  # unique element, no cache
            return self.___inverse_metric_matrix___(*xi_et_sg)
        else:
            cached, data = ndarray_key_comparer(___cache_msehtt_imm___, xi_et_sg, check_str=self._metric_signature)
            if cached:
                pass
            else:
                data = self.___inverse_metric_matrix___(*xi_et_sg)
                add_to_ndarray_cache(___cache_msehtt_imm___, xi_et_sg, data, check_str=self._metric_signature)
            return data

    def ___inverse_metric_matrix___(self, *xi_et_sg):
        """"""
        ijm = self.inverse_Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m, self._element.n
        iG = [[None for _ in range(m)] for __ in range(m)]
        for i in range(m):
            for j in range(i, m):
                # noinspection PyTypeChecker
                iG[i][j] = ijm[i][0] * ijm[j][0]
                for L in range(1, n):
                    iG[i][j] += ijm[i][L] * ijm[j][L]
                if i != j:
                    iG[j][i] = iG[i][j]
        return iG


# ======== FACE CT BASE =============================================================================
class _FaceCoordinateTransformationBase(Frozen):
    """"""
    def __init__(self, face):
        self._element = face._element
        self._face = face
        self.___c_ounv___ = None
        self._freeze()

    def mapping(self, *xi_et):
        """"""
        raise NotImplementedError()

    def Jacobian_matrix(self, *xi_et):
        """"""
        raise NotImplementedError()

    def outward_unit_normal_vector(self, *xi_et):
        """The outward unit norm vector (vec{n})."""
        raise NotImplementedError()

    def is_plane(self):
        """Return True if the face is plane; a straight line in 2d space or a plane surface in 3d space for example."""
        raise NotImplementedError()

    @property
    def constant_outward_unit_normal_vector(self):
        """"""
        if self.is_plane():
            if self.___c_ounv___ is None:
                if self._element.m == self._element.n == 2:
                    c_ounv = self.outward_unit_normal_vector(np.array([0]))
                    if isinstance(c_ounv[0], np.ndarray):
                        c_ounv0 = c_ounv[0][0]
                    else:
                        c_ounv0 = c_ounv[0]
                    if isinstance(c_ounv[1], np.ndarray):
                        c_ounv1 = c_ounv[1][0]
                    else:
                        c_ounv1 = c_ounv[1]
                    self.___c_ounv___ = (c_ounv0, c_ounv1)
                else:
                    raise NotImplementedError()

            return self.___c_ounv___

        else:
            raise Exception(f'Face {self._face} is not plane. Thus it has no constant_outward_unit_normal_vector!')
