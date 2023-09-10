# -*- coding: utf-8 -*-
r"""
"""
from typing import Dict
import numpy as np

from tools.frozen import Frozen


class MseHyPy2SpaceReconstructLambda(Frozen):
    """Reconstruct over all mesh-elements."""

    def __init__(self, space):
        """"""
        self._mesh = space.mesh
        self._space = space
        self._freeze()

    def __call__(self, generation, cochain, *meshgrid, **kwargs):
        """Reconstruct using cochain at time `t` on the mesh grid of `meshgrid_xi_et_sg`."""
        generation = self._space._pg(generation)
        abs_sp = self._space.abstract
        k = abs_sp.k
        orientation = abs_sp.orientation

        if k == 1:
            return getattr(self, f'_k{k}_{orientation}')(
                generation, cochain, *meshgrid, **kwargs)
        else:
            return getattr(self, f'_k{k}')(
                generation, cochain, *meshgrid, **kwargs)

    def _k0(self, generation, cochain, *meshgrid_xi_et, ravel=False):
        """"""
        degree = cochain._f.degree
        representative = self._mesh[generation]
        local_cochain = cochain.local

        xi, et = meshgrid_xi_et
        shape: list = [len(xi), len(et)]
        xi_et, bf_qt = self._space.basis_functions(degree, generation, *meshgrid_xi_et)

        xy = representative.ct.mapping(*xi_et)

        x_dict: Dict = dict()
        y_dict: Dict = dict()
        v_dict: Dict = dict()
        for e in xy:  # go through all fundamental cells.
            x, y = xy[e]
            x_dict[e] = x
            y_dict[e] = y
            bf = bf_qt[e][0]

            lce = local_cochain[e]
            v = np.einsum('ij, i -> j', bf, lce, optimize='optimal')
            v_dict[e] = v

        if ravel:
            pass
        else:
            for e in x_dict:
                x_dict[e] = x_dict[e].reshape(shape, order='F')
                y_dict[e] = y_dict[e].reshape(shape, order='F')
                v_dict[e] = v_dict[e].reshape(shape, order='F')

        return (x_dict, y_dict), (v_dict, )

    def _k1_inner(self, generation, cochain, *meshgrid_xi_et, ravel=False):
        """"""
        degree = cochain._f.degree
        representative = self._mesh[generation]
        local_cochain = cochain.local

        xi, et = meshgrid_xi_et
        shape: list = [len(xi), len(et)]
        xi_et, bf_qt = self._space.basis_functions(degree, generation, *meshgrid_xi_et)

        num_components_qt = self._space.num_local_dof_components(degree)

        xy = representative.ct.mapping(*xi_et)
        iJ = representative.ct.inverse_Jacobian_matrix(*xi_et)

        x_dict: Dict = dict()
        y_dict: Dict = dict()
        v0_dict: Dict = dict()
        v1_dict: Dict = dict()
        for e in xy:  # go through all fundamental cells.
            x, y = xy[e]
            x_dict[e] = x
            y_dict[e] = y

            fc = representative[e]
            bf = bf_qt[e]
            lce = local_cochain[e]

            local_0 = lce[:num_components_qt[fc._type][0]]
            local_1 = lce[num_components_qt[fc._type][0]:]

            u = np.einsum('ij, i -> j', bf[0], local_0, optimize='optimal')
            v = np.einsum('ij, i -> j', bf[1], local_1, optimize='optimal')

            iJe = iJ[e]
            iJ0, iJ1 = iJe
            iJ00, iJ01 = iJ0
            iJ10, iJ11 = iJ1

            if not isinstance(iJ10, np.ndarray) and iJ10 == 0:
                v0 = u * iJ00
            else:
                v0 = u * iJ00 + v * iJ10

            if not isinstance(iJ01, np.ndarray) and iJ01 == 0:
                v1 = v * iJ11
            else:
                v1 = u * iJ01 + v * iJ11

            v0_dict[e] = v0
            v1_dict[e] = v1

        if ravel:
            pass
        else:
            for e in x_dict:
                x_dict[e] = x_dict[e].reshape(shape, order='F')
                y_dict[e] = y_dict[e].reshape(shape, order='F')
                v0_dict[e] = v0_dict[e].reshape(shape, order='F')
                v1_dict[e] = v1_dict[e].reshape(shape, order='F')

        return (x_dict, y_dict), (v0_dict, v1_dict)

    def _k1_outer(self, generation, cochain, *meshgrid_xi_et, ravel=False):
        """"""
        degree = cochain._f.degree
        representative = self._mesh[generation]
        local_cochain = cochain.local

        xi, et = meshgrid_xi_et
        shape: list = [len(xi), len(et)]
        xi_et, bf_qt = self._space.basis_functions(degree, generation, *meshgrid_xi_et)

        num_components_qt = self._space.num_local_dof_components(degree)

        xy = representative.ct.mapping(*xi_et)
        iJ = representative.ct.inverse_Jacobian_matrix(*xi_et)

        x_dict: Dict = dict()
        y_dict: Dict = dict()
        v0_dict: Dict = dict()
        v1_dict: Dict = dict()
        for e in xy:  # go through all fundamental cells.
            x, y = xy[e]
            x_dict[e] = x
            y_dict[e] = y

            fc = representative[e]
            bf = bf_qt[e]
            lce = local_cochain[e]

            local_0 = lce[:num_components_qt[fc._type][0]]
            local_1 = lce[num_components_qt[fc._type][0]:]

            u = np.einsum('ij, i -> j', bf[0], local_0, optimize='optimal')
            v = np.einsum('ij, i -> j', bf[1], local_1, optimize='optimal')

            iJe = iJ[e]
            iJ0, iJ1 = iJe
            iJ00, iJ01 = iJ0
            iJ10, iJ11 = iJ1

            if not isinstance(iJ01, np.ndarray) and iJ01 == 0:
                v0 = + u * iJ11
            else:
                v0 = + u * iJ11 - v * iJ01

            if not isinstance(iJ10, np.ndarray) and iJ10 == 0:
                v1 = + v * iJ00
            else:
                v1 = - u * iJ10 + v * iJ00

            v0_dict[e] = v0
            v1_dict[e] = v1

        if ravel:
            pass
        else:
            for e in x_dict:
                x_dict[e] = x_dict[e].reshape(shape, order='F')
                y_dict[e] = y_dict[e].reshape(shape, order='F')
                v0_dict[e] = v0_dict[e].reshape(shape, order='F')
                v1_dict[e] = v1_dict[e].reshape(shape, order='F')

        return (x_dict, y_dict), (v0_dict, v1_dict)

    def _k2(self, generation, cochain, *meshgrid_xi_et, ravel=False):
        """"""
        degree = cochain._f.degree
        representative = self._mesh[generation]
        local_cochain = cochain.local

        xi, et = meshgrid_xi_et
        shape: list = [len(xi), len(et)]
        xi_et, bf_qt = self._space.basis_functions(degree, generation, *meshgrid_xi_et)
        xy = representative.ct.mapping(*xi_et)
        iJ = representative.ct.inverse_Jacobian(*xi_et)

        x_dict: Dict = dict()
        y_dict: Dict = dict()
        v_dict: Dict = dict()
        for e in xy:  # go through all fundamental cells.
            x, y = xy[e]
            x_dict[e] = x
            y_dict[e] = y
            bf = bf_qt[e][0]
            iJe = iJ[e]
            lce = local_cochain[e]
            v = np.einsum(
                'i, j, ij -> j',
                lce, iJe, bf,
                optimize='optimal',
            )
            v_dict[e] = v

        if ravel:
            pass
        else:
            for e in x_dict:
                x_dict[e] = x_dict[e].reshape(shape, order='F')
                y_dict[e] = y_dict[e].reshape(shape, order='F')
                v_dict[e] = v_dict[e].reshape(shape, order='F')

        return (x_dict, y_dict), (v_dict, )
