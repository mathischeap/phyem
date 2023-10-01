# -*- coding: utf-8 -*-
r"""
"""

import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class NormLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._mesh = space.mesh
        self._space = space
        self._freeze()

    def __call__(self, cochain, d=2):
        """default: L^d norm"""
        return self.L(cochain, d=d)

    def L(self, cochain, d=2):
        """compute the L^d norm of the root-form"""
        abs_sp = self._space.abstract
        k = abs_sp.k
        assert isinstance(d, int) and d > 0, f"d={d} is wrong."

        degree = cochain._f.degree
        p = self._space[degree].p

        quad_degree = [p + 2, p + 2]

        return getattr(self, f'_L_norm_k{k}')(d, quad_degree, cochain)

    def _L_norm_k2(self, d, quad_degree, cochain):
        """m2 n2 k2"""
        return self._L_norm_k0(d, quad_degree, cochain)

    def _L_norm_k1(self, d, quad_degree, cochain):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad
        xy, v = self._space.reconstruct(cochain, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        u, v = v
        integral = list()
        for e in v:
            dis_u = u[e]
            dis_v = v[e]
            metric = J[e]
            dis = dis_u ** d + dis_v ** d
            integral.append(
                np.einsum('ij, i, j -> ', dis * metric, *quad_weights, optimize='optimal')
            )
        return np.sum(integral) ** (1/d)

    def _L_norm_k0(self, d, quad_degree, cochain):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad
        xy, v = self._space.reconstruct(cochain, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        v = v[0]
        integral = list()
        for e in v:
            dis_v = v[e]
            metric = J[e]
            integral.append(
                np.einsum('ij, i, j -> ', dis_v * metric, *quad_weights, optimize='optimal')
            )
        return np.sum(integral) ** (1/d)
