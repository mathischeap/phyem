# -*- coding: utf-8 -*-
r"""
"""
from typing import Dict
import numpy as np

from tools.frozen import Frozen


class ReconstructLambda(Frozen):
    """Reconstruct over all mesh-elements."""

    def __init__(self, space):
        """"""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._freeze()

    def __call__(self, cochain, meshgrid_xi, meshgrid_et, ravel, element_range=None, degree=None):
        """Reconstruct using cochain at time `t` on the mesh grid of `meshgrid_xi_et_sg`."""

        k = self._k
        orientation = self._orientation

        if k == 1:
            return getattr(self, f'_k{k}_{orientation}')(
                cochain, meshgrid_xi, meshgrid_et, ravel, element_range=element_range, degree=degree)
        else:
            return getattr(self, f'_k{k}')(
                cochain, meshgrid_xi, meshgrid_et, ravel, element_range=element_range, degree=degree)

    def _coordinates_only(self, xi, et, ravel, element_range=None):
        """"""
        shape: list = [len(xi), len(et)]

        xi, et = np.meshgrid(xi, et, indexing='ij')
        xi, et = (xi.ravel('F'), et.ravel('F'))
        xy = self._mesh.ct.mapping(xi, et, element_range=element_range)

        x_dict: Dict = dict()
        y_dict: Dict = dict()
        for index in xy:  # go through all elements
            x, y = xy[index]
            x_dict[index] = x
            y_dict[index] = y

        if ravel:
            pass
        else:
            for e in x_dict:
                x_dict[e] = x_dict[e].reshape(shape, order='F')
                y_dict[e] = y_dict[e].reshape(shape, order='F')

        return x_dict, y_dict

    def _k0(self, cochain, xi, et, ravel, element_range=None, degree=None):
        """"""
        if degree is None:
            degree = cochain._f.degree
        else:
            pass
        assert degree is not None, f"must have a degree"

        if isinstance(cochain, dict):
            local_cochain = cochain
        else:
            local_cochain = cochain.local

        shape: list = [len(xi), len(et)]
        xi_et, bf_qt = self._space.basis_functions(degree, xi, et)

        xy = self._mesh.ct.mapping(*xi_et, element_range=element_range)

        x_dict: Dict = dict()
        y_dict: Dict = dict()
        v_dict: Dict = dict()
        for index in xy:  # go through all elements
            x, y = xy[index]
            x_dict[index] = x
            y_dict[index] = y
            bf = bf_qt[index][0]

            lce = local_cochain[index]
            v = np.einsum('ij, i -> j', bf, lce, optimize='optimal')
            v_dict[index] = v

        if ravel:
            pass
        else:
            for e in x_dict:
                x_dict[e] = x_dict[e].reshape(shape, order='F')
                y_dict[e] = y_dict[e].reshape(shape, order='F')
                v_dict[e] = v_dict[e].reshape(shape, order='F')

        return (x_dict, y_dict), (v_dict, )

    def _k1_inner(self, cochain, xi, et, ravel, element_range=None, degree=None):
        """"""
        if degree is None:
            degree = cochain._f.degree
        else:
            pass
        assert degree is not None, f"must have a degree"

        if isinstance(cochain, dict):
            local_cochain = cochain
        else:
            local_cochain = cochain.local

        shape: list = [len(xi), len(et)]
        xi_et, bf_qt = self._space.basis_functions(degree, xi, et)

        num_components_qt = self._space.num_local_dof_components(degree)

        xy = self._mesh.ct.mapping(*xi_et, element_range=element_range)
        iJ = self._mesh.ct.inverse_Jacobian_matrix(*xi_et, element_range=element_range)

        x_dict: Dict = dict()
        y_dict: Dict = dict()
        v0_dict: Dict = dict()
        v1_dict: Dict = dict()
        for index in xy:  # go through all elements
            x, y = xy[index]
            x_dict[index] = x
            y_dict[index] = y

            ele_type = self._mesh[index].type

            bf = bf_qt[index]
            lce = local_cochain[index]

            local_0 = lce[:num_components_qt[ele_type][0]]
            local_1 = lce[num_components_qt[ele_type][0]:]

            u = np.einsum('ij, i -> j', bf[0], local_0, optimize='optimal')
            v = np.einsum('ij, i -> j', bf[1], local_1, optimize='optimal')

            iJe = iJ[index]
            iJ0, iJ1 = iJe
            iJ00, iJ01 = iJ0
            iJ10, iJ11 = iJ1

            v0 = u * iJ00 + v * iJ10
            v1 = u * iJ01 + v * iJ11

            v0_dict[index] = v0
            v1_dict[index] = v1

        if ravel:
            pass
        else:
            for e in x_dict:
                x_dict[e] = x_dict[e].reshape(shape, order='F')
                y_dict[e] = y_dict[e].reshape(shape, order='F')
                v0_dict[e] = v0_dict[e].reshape(shape, order='F')
                v1_dict[e] = v1_dict[e].reshape(shape, order='F')

        return (x_dict, y_dict), (v0_dict, v1_dict)

    def _k1_outer(self, cochain, xi, et, ravel, element_range=None, degree=None):
        """"""
        if degree is None:
            degree = cochain._f.degree
        else:
            pass
        assert degree is not None, f"must have a degree"

        if isinstance(cochain, dict):
            local_cochain = cochain
        else:
            local_cochain = cochain.local

        shape: list = [len(xi), len(et)]
        xi_et, bf_qt = self._space.basis_functions(degree, xi, et)

        num_components_qt = self._space.num_local_dof_components(degree)

        xy = self._mesh.ct.mapping(*xi_et, element_range=element_range)
        iJ = self._mesh.ct.inverse_Jacobian_matrix(*xi_et, element_range=element_range)

        x_dict: Dict = dict()
        y_dict: Dict = dict()
        v0_dict: Dict = dict()
        v1_dict: Dict = dict()
        for index in xy:  # go through all elements
            x, y = xy[index]
            x_dict[index] = x
            y_dict[index] = y

            ele_type = self._mesh[index].type

            bf = bf_qt[index]
            lce = local_cochain[index]

            local_0 = lce[:num_components_qt[ele_type][0]]
            local_1 = lce[num_components_qt[ele_type][0]:]

            u = np.einsum('ij, i -> j', bf[0], local_0, optimize='optimal')
            v = np.einsum('ij, i -> j', bf[1], local_1, optimize='optimal')

            iJe = iJ[index]
            iJ0, iJ1 = iJe
            iJ00, iJ01 = iJ0
            iJ10, iJ11 = iJ1

            v0 = + u * iJ11 - v * iJ01
            v1 = - u * iJ10 + v * iJ00

            v0_dict[index] = v0
            v1_dict[index] = v1

        if ravel:
            pass
        else:
            for e in x_dict:
                x_dict[e] = x_dict[e].reshape(shape, order='F')
                y_dict[e] = y_dict[e].reshape(shape, order='F')
                v0_dict[e] = v0_dict[e].reshape(shape, order='F')
                v1_dict[e] = v1_dict[e].reshape(shape, order='F')

        return (x_dict, y_dict), (v0_dict, v1_dict)

    def _k2(self, cochain, xi, et, ravel, element_range=None, degree=None):
        """"""
        if degree is None:
            degree = cochain._f.degree
        else:
            pass
        assert degree is not None, f"must have a degree"

        if isinstance(cochain, dict):
            local_cochain = cochain
        else:
            local_cochain = cochain.local

        shape: list = [len(xi), len(et)]
        xi_et, bf_qt = self._space.basis_functions(degree, xi, et)
        xy = self._mesh.ct.mapping(*xi_et, element_range=element_range)
        iJ = self._mesh.ct.inverse_Jacobian(*xi_et, element_range=element_range)

        x_dict: Dict = dict()
        y_dict: Dict = dict()
        v_dict: Dict = dict()
        for index in xy:  # go through all valid elements.
            x, y = xy[index]
            x_dict[index] = x
            y_dict[index] = y
            bf = bf_qt[index][0]

            iJe = iJ[index]
            lce = local_cochain[index]

            v = np.einsum(
                'i, j, ij -> j',
                lce, iJe, bf,
                optimize='optimal',
            )
            v_dict[index] = v

        if ravel:
            pass
        else:
            for e in x_dict:
                x_dict[e] = x_dict[e].reshape(shape, order='F')
                y_dict[e] = y_dict[e].reshape(shape, order='F')
                v_dict[e] = v_dict[e].reshape(shape, order='F')

        return (x_dict, y_dict), (v_dict, )
