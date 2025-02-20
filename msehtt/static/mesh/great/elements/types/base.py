# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from src.spaces.main import _degree_str_maker

from msehtt.static.space.basis_function.Lambda.bf_m2n2k0 import ___bf220_msepy_quadrilateral___
from msehtt.static.space.basis_function.Lambda.bf_m2n2k1 import ___bf221o_outer_msepy_quadrilateral___
from msehtt.static.space.basis_function.Lambda.bf_m2n2k1 import ___bf221i_inner_msepy_quadrilateral___
from msehtt.static.space.basis_function.Lambda.bf_m2n2k2 import ___bf222_msepy_quadrilateral___

from msehtt.static.space.basis_function.Lambda.bf_m2n2k0 import ___bf220_utv_5_triangle___
from msehtt.static.space.basis_function.Lambda.bf_m2n2k1 import ___bf221i_inner_vtu_5___
from msehtt.static.space.basis_function.Lambda.bf_m2n2k1 import ___bf221o_outer_vtu_5___

from msehtt.static.space.basis_function.Lambda.bf_m3n3k0 import ___bf330_msepy_quadrilateral___
from msehtt.static.space.basis_function.Lambda.bf_m3n3k1 import ___bf331_msepy_quadrilateral___
from msehtt.static.space.basis_function.Lambda.bf_m3n3k2 import ___bf332_msepy_quadrilateral___
from msehtt.static.space.basis_function.Lambda.bf_m3n3k3 import ___bf333_msepy_quadrilateral___

from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import local_numbering_Lambda__m2n2k1_outer
from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import local_numbering_Lambda__m2n2k1_inner

from msehtt.static.space.find.local_dofs_on_face.Lambda.m2n2k1 import find_local_dofs_on_face__m2n2k1_outer
from msehtt.static.space.find.local_dofs_on_face.Lambda.m2n2k1 import find_local_dofs_on_face__m2n2k1_inner
from msehtt.static.space.find.local_dofs_on_face.Lambda.m2n2k0 import find_local_dofs_on_face__m2n2k0

from msehtt.static.space.find.local_dofs_on_face.Lambda.m3n3k0 import find_local_dofs_on_face__m3n3k0
from msehtt.static.space.find.local_dofs_on_face.Lambda.m3n3k1 import find_local_dofs_on_face__m3n3k1
from msehtt.static.space.find.local_dofs_on_face.Lambda.m3n3k2 import find_local_dofs_on_face__m3n3k2


___xi___ = np.linspace(-1, 1, 3)
___et___ = np.linspace(-1, 1, 3)
___xi2___, ___et2___ = np.meshgrid(___xi___, ___et___, indexing='ij')
___xi2___ = ___xi2___.ravel('F')
___et2___ = ___et2___.ravel('F')
___xi___ = np.linspace(-1, 1, 3)
___et___ = np.linspace(-1, 1, 3)
___sg___ = np.linspace(-1, 1, 3)
___xi3___, ___et3___, ___sg3___ = np.meshgrid(___xi___, ___et___, ___sg___, indexing='ij')
___xi3___ = ___xi3___.ravel('F')
___et3___ = ___et3___.ravel('F')
___sg3___ = ___sg3___.ravel('F')


___degree_cache_pool___ = {}


class MseHttGreatMeshBaseElement(Frozen):
    r""""""

    def __init__(self):
        r""""""
        self._index = None
        self._map = None
        self._parameters = None
        self._ct = None
        self._faces = None
        self._edges = None
        self._dof_reverse_info = {}
        self._freeze()

    @property
    def index(self):
        r"""The index of this element. Must be unique all over the great mesh."""
        return self._index

    @classmethod
    def m(cls):
        r"""the dimensions of the space"""
        raise NotImplementedError()

    @classmethod
    def n(cls):
        r"""the dimensions of the element"""
        raise NotImplementedError()

    @classmethod
    def _find_element_center_coo(cls, parameters):
        r""""""
        raise NotImplementedError(f"`_find_element_center_coo` method not coded for {cls}.")

    @classmethod
    def _find_mapping_(cls, parameters, x, y):
        r"""A class method that compute ct.mapping with parameters. With this, we can do
        some checks before the element is actually made.
        """
        raise NotImplementedError()

    @property
    def etype(self):
        r"""Return the indicator of the element type."""
        return self._etype()

    @classmethod
    def _etype(cls):
        r""""""
        raise NotImplementedError()

    @property
    def parameters(self):
        r""""""
        return self._parameters

    @property
    def metric_signature(self):
        r""""""
        raise NotImplementedError()

    @property
    def signature(self):
        r"""Each element has its unique signature. Even two elements have same metric, their signature are different.
        This is mainly used to read data. For example, when we read cochain from a file, we need to use signatures of
        elements to make sure we are reading the correct data into the correct elements because sometimes, elements
        may be numbered differently.
        """
        if self.m() == self.n() == 2:  # 2d element in 2d space
            X, Y = self.ct.mapping(___xi2___, ___et2___)
            signature = list()
            for x, y in zip(X, Y):
                signature.append("({:.4f},{:.4f})".format(x, y))
            signature = ''.join(signature)
        elif self.m() == self.n() == 3:  # 3d element in 3d space
            X, Y, Z = self.ct.mapping(___xi3___, ___et3___, ___sg3___)
            signature = list()
            for x, y, z in zip(X, Y, Z):
                signature.append("({:.3f},{:.3f},{:.3f})".format(x, y, z))
            signature = ''.join(signature)
        else:
            raise NotImplementedError((self.m(), self.n()))

        return str(self.etype) + signature

    @property
    def map_(self):
        r"""The numbering of the nodes of the element.

        For example, for a rectangle element, element.map_ = [0, 100, 54 33]. This means the four nodes of
        this element are labeled 0, 100, 54 33 respectively.
        """
        return self._map

    def local_numbering(self, indicator, degree):
        r""""""
        p = self.degree_parser(degree)[0]
        if indicator == 'm2n2k1_outer':
            return local_numbering_Lambda__m2n2k1_outer(self.etype, p)
        elif indicator == 'm2n2k1_inner':
            return local_numbering_Lambda__m2n2k1_inner(self.etype, p)
        else:
            raise NotImplementedError()

    def find_local_dofs_on_face(self, indicator, degree, face_index, component_wise=False):
        r""""""
        p = self.degree_parser(degree)[0]
        if indicator == 'm2n2k1_outer':
            return find_local_dofs_on_face__m2n2k1_outer(self.etype, p, face_index, component_wise=component_wise)
        elif indicator == 'm2n2k1_inner':
            return find_local_dofs_on_face__m2n2k1_inner(self.etype, p, face_index, component_wise=component_wise)
        elif indicator == 'm2n2k0':
            return find_local_dofs_on_face__m2n2k0(self.etype, p, face_index)
        elif indicator == 'm3n3k0':
            return find_local_dofs_on_face__m3n3k0(self.etype, p, face_index)
        elif indicator == 'm3n3k1':
            return find_local_dofs_on_face__m3n3k1(self.etype, p, face_index, component_wise=component_wise)
        elif indicator == 'm3n3k2':
            return find_local_dofs_on_face__m3n3k2(self.etype, p, face_index, component_wise=component_wise)
        else:
            raise NotImplementedError(indicator)

    def bf(self, indicator, degree, *grid_mesh):
        r""""""
        p, btype = self.degree_parser(degree)

        if self.etype in (
                9,
                'orthogonal rectangle',
                'unique msepy curvilinear quadrilateral',
                'unique curvilinear quad',
        ):
            if indicator == 'm2n2k0':
                xi_et_sg, bf = ___bf220_msepy_quadrilateral___(p, btype, *grid_mesh)
            elif indicator == 'm2n2k1_outer':
                xi_et_sg, bf = ___bf221o_outer_msepy_quadrilateral___(p, btype, *grid_mesh)
            elif indicator == 'm2n2k1_inner':
                xi_et_sg, bf = ___bf221i_inner_msepy_quadrilateral___(p, btype, *grid_mesh)
            elif indicator == 'm2n2k2':
                xi_et_sg, bf = ___bf222_msepy_quadrilateral___(p, btype, *grid_mesh)
            else:
                raise NotImplementedError()

        elif self.etype in (
                5,
                "unique msepy curvilinear triangle"
        ):
            if indicator == 'm2n2k0':
                xi_et_sg, bf = ___bf220_utv_5_triangle___(p, btype, *grid_mesh)
            elif indicator == 'm2n2k1_outer':
                xi_et_sg, bf = ___bf221o_outer_vtu_5___(p, btype, *grid_mesh)
            elif indicator == 'm2n2k1_inner':
                xi_et_sg, bf = ___bf221i_inner_vtu_5___(p, btype, *grid_mesh)
            elif indicator == 'm2n2k2':
                xi_et_sg, bf = ___bf222_msepy_quadrilateral___(p, btype, *grid_mesh)  # same as in a rectangle
            else:
                raise NotImplementedError()

        elif self.etype in ('orthogonal hexahedron', ):
            if indicator == 'm3n3k0':
                xi_et_sg, bf = ___bf330_msepy_quadrilateral___(p, btype, *grid_mesh)
            elif indicator == 'm3n3k1':
                xi_et_sg, bf = ___bf331_msepy_quadrilateral___(p, btype, *grid_mesh)
            elif indicator == 'm3n3k2':
                xi_et_sg, bf = ___bf332_msepy_quadrilateral___(p, btype, *grid_mesh)
            elif indicator == 'm3n3k3':
                xi_et_sg, bf = ___bf333_msepy_quadrilateral___(p, btype, *grid_mesh)
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError(f"bf not implemented for etype={self.etype}")

        if self._dof_reverse_info == {}:
            return xi_et_sg, bf
        else:
            pass

        if indicator in self._dof_reverse_info:
            if indicator == 'm2n2k1_outer':
                bf0 = bf[0].copy()
                bf1 = bf[1].copy()
                face_indices = self.dof_reverse_info['m2n2k1_outer']
                for fi in face_indices:
                    component, local_dofs = find_local_dofs_on_face__m2n2k1_outer(
                        self.etype, p, fi, component_wise=True)
                    if component == 0:
                        bf0[local_dofs, :] = - bf[0][local_dofs, :]
                    elif component == 1:
                        bf1[local_dofs, :] = - bf[1][local_dofs, :]
                    else:
                        raise Exception()
                return xi_et_sg, [bf0, bf1]

            elif indicator == 'm2n2k1_inner':
                bf0 = bf[0].copy()
                bf1 = bf[1].copy()
                face_indices = self.dof_reverse_info['m2n2k1_inner']
                for fi in face_indices:
                    component, local_dofs = find_local_dofs_on_face__m2n2k1_inner(
                        self.etype, p, fi, component_wise=True)
                    if component == 0:
                        bf0[local_dofs, :] = - bf[0][local_dofs, :]
                    elif component == 1:
                        bf1[local_dofs, :] = - bf[1][local_dofs, :]
                    else:
                        raise Exception()
                return xi_et_sg, [bf0, bf1]

            else:
                raise NotImplementedError(f"reverse dof not implemented for {indicator}-form")

        else:
            return xi_et_sg, bf

    @property
    def dof_reverse_info(self):
        r"""

        Returns
        -------
        dof_reverse_info : dict
            For example:
                {
                    'm2n2k1_inner': [0],  # face id, dofs on this face need a minus.
                    'm2n2k1_outer': [2, 3],  # face id, dofs on these faces need a minus.
                }

        """
        return self._dof_reverse_info

    @property
    def ct(self):
        r""""""
        return self._ct

    def _generate_outline_data(self, ddf=1):
        r""""""
        raise NotImplementedError()

    def _find_element_quality(self):
        r"""Return a factor indicating the quality of the elements.

        When the factor is 0: the element is worst.
        When the factor is 1: the element is best.

        Returns
        -------
        quality: float
            In [0, 1]. 0: Worst; 1: Best.

        """
        raise NotImplementedError(f"quality not implemented for element type: {self.etype}.")

    @classmethod
    def face_setting(cls):
        r"""The face setting; show nodes of each face of the element and the positive direction.

        A face can be 2-d (face of 3d element) or 1-d (face of 2d element).
        """
        raise NotImplementedError()

    @classmethod
    def edge_setting(cls):
        r"""The edge setting; show nodes of each edge of the element and the positive direction.

        An edge can be 1-d for 3d element. 2d element has no edge (only have face.)
        """
        raise NotImplementedError()

    def ___face_representative_str___(self):
        r""""""
        raise NotImplementedError()

    def ___edge_representative_str___(self):
        r""""""
        raise NotImplementedError()

    @property
    def faces(self):
        r""""""
        raise NotImplementedError()

    @property
    def edges(self):
        r""""""
        return NotImplementedError()

    @classmethod
    def degree_parser(cls, degree):
        r""""""
        key = cls.__name__ + ':' + _degree_str_maker(degree)
        if key in ___degree_cache_pool___:
            return ___degree_cache_pool___[key]
        else:
            if isinstance(degree, int):
                assert degree >= 1, f'Must be'
                if cls.m() == cls.n() == 2:
                    p = (degree, degree)
                elif cls.m() == cls.n() == 3:
                    p = (degree, degree, degree)
                else:
                    raise NotImplementedError()
                dtype = 'Lobatto'
            else:
                raise NotImplementedError()
            ___degree_cache_pool___[key] = p, dtype
            return p, dtype

    # @classmethod
    # def _form_face_dof_direction_topology(cls):
    #     """"""
    #     return None

    def _generate_element_vtk_data_(self, *args):
        r""""""
        raise NotImplementedError()

    def _generate_vtk_data_for_form(self, indicator, element_cochain, degree, data_density):
        r""""""
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
    r""""""
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
    r""""""
    def __init__(self, element, metric_signature):
        r""""""
        self._element = element
        self._metric_signature = metric_signature
        self._freeze()

    def __repr__(self):
        r""""""
        return f"<CT of {self._element.__repr__()}>"

    def mapping(self, *xi_et_sg):
        r""""""
        raise NotImplementedError()

    def ___Jacobian_matrix___(self, *xi_et_sg):
        r""""""
        raise NotImplementedError()

    def Jacobian_matrix(self, *xi_et_sg):
        r""""""
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
        r""""""
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
        r""""""
        jm = self.Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m(), self._element.n()

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
        r"""the Determinant of the Jacobian matrix. When Jacobian matrix is square, Jacobian = sqrt(g)."""
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
        r""""""
        jm = self.Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m(), self._element.n()

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
        r"""For square Jacobian matrix,
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
        r""""""
        m, n = self._element.m(), self._element.n()
        if m == n:
            return self.Jacobian(*xi_et_sg) ** 2
        else:
            raise NotImplementedError()

    def inverse_Jacobian(self, *xi_et_sg):
        r"""the Determinant of the inverse Jacobian matrix."""
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
        r""""""
        ijm = self.inverse_Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m(), self._element.n()

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
        r""""""
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
        r""""""
        jm = self.Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m(), self._element.n()
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
        r""""""
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
        r""""""
        ijm = self.inverse_Jacobian_matrix(*xi_et_sg)
        m, n = self._element.m(), self._element.n()
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
    r""""""
    def __init__(self, face):
        r""""""
        self._element = face._element
        self._face = face
        self.___c_ounv___ = None
        self._freeze()

    def mapping(self, *xi_et):
        r""""""
        raise NotImplementedError()

    def Jacobian_matrix(self, *xi_et):
        r""""""
        raise NotImplementedError()

    def outward_unit_normal_vector(self, *xi_et_sg):
        r"""The outward unit norm vector (vec{n})."""
        raise NotImplementedError()

    def is_plane(self):
        r"""Return True if the face is plane; a straight line in 2d space or a plane surface in 3d space for example."""
        raise NotImplementedError()

    @property
    def constant_outward_unit_normal_vector(self):
        r""""""
        if self.is_plane():
            if self.___c_ounv___ is None:
                if self._element.m() == self._element.n() == 2:  # mn=(2, 2); 2d element in 2d space
                    c_ounv = self.outward_unit_normal_vector(np.array([0]))
                    if isinstance(c_ounv[0], np.ndarray):
                        c_ounv0 = c_ounv[0][0]
                    else:
                        c_ounv0 = c_ounv[0]
                    if isinstance(c_ounv[1], np.ndarray):
                        c_ounv1 = c_ounv[1][0]
                    else:
                        c_ounv1 = c_ounv[1]
                    c_ounv0 = round(c_ounv0, 8)  # remove the round-off error
                    c_ounv1 = round(c_ounv1, 8)  # remove the round-off error
                    self.___c_ounv___ = (c_ounv0, c_ounv1)

                elif self._element.m() == self._element.n() == 3:  # mn=(3, 3); 3d element in 3d space
                    c_ounv = self.outward_unit_normal_vector(np.array([0]), np.array([0]), np.array([0]))
                    if isinstance(c_ounv[0], np.ndarray):
                        c_ounv0 = c_ounv[0][0]
                    else:
                        c_ounv0 = c_ounv[0]
                    if isinstance(c_ounv[1], np.ndarray):
                        c_ounv1 = c_ounv[1][0]
                    else:
                        c_ounv1 = c_ounv[1]
                    if isinstance(c_ounv[2], np.ndarray):
                        c_ounv2 = c_ounv[2][0]
                    else:
                        c_ounv2 = c_ounv[2]
                    c_ounv0 = round(c_ounv0, 8)  # remove the round-off error
                    c_ounv1 = round(c_ounv1, 8)  # remove the round-off error
                    c_ounv2 = round(c_ounv2, 8)  # remove the round-off error
                    self.___c_ounv___ = (c_ounv0, c_ounv1, c_ounv2)

                else:
                    raise NotImplementedError()

            return self.___c_ounv___

        else:
            raise Exception(
                f'Face {self._face} is not plane. '
                f'Thus it has no constant_outward_unit_normal_vector!'
            )
