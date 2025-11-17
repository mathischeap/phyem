# -*- coding: utf-8 -*-
r"""
"""

import numpy as np

from phyem.tools.frozen import Frozen
from phyem.tools.quadrature import Quadrature


class MsePySpaceNormLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._mesh = space.mesh
        self._space = space
        self._freeze()

    def __call__(self, local_cochain, degree, quad_degree=None, d=2):
        """default: L^d norm"""
        return self.L(local_cochain, degree, quad_degree=quad_degree, d=d)

    def L(self, local_cochain, degree, quad_degree=None, d=2):
        """compute the L^d norm of the root-form"""
        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        assert isinstance(d, int) and d > 0, f"d={d} is wrong."

        if quad_degree is None:
            quad_degree = [i + 1 for i in self._space[degree].p]  # + 1
        else:
            pass

        return getattr(self, f'_L_norm_m{m}_n{n}_k{k}')(d, quad_degree, local_cochain, degree)

    def _L_norm_m1_n1_k0(self, d, quad_degree, local_cochain, degree):
        """"""
        return self._L_norm_m1_n1_k1(d, quad_degree, local_cochain, degree)

    def _L_norm_m1_n1_k1(self, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._space.reconstruct(local_cochain, degree, quad_nodes)
        J = self._mesh.ct.Jacobian(quad_nodes)
        v = v[0]

        integral = list()
        for ri in self._mesh.regions:
            start, end = self._mesh.elements._elements_in_region(ri)
            dis_v = v[start:end, :]
            metric = J(range(start, end))
            integral.append(
                np.einsum(
                    'ij -> ',
                    dis_v ** d * metric * quad_weights,
                    optimize='optimal'
                )
            )

        return np.sum(integral) ** (1/d)

    def _L_norm_m2_n2_k2(self, d, quad_degree, local_cochain, degree):
        """m2 n2 k2"""
        return self._L_norm_m2_n2_k0(d, quad_degree, local_cochain, degree)

    def _L_norm_m2_n2_k1(self, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        u, v = v
        integral = list()

        for ri in self._mesh.regions:
            start, end = self._mesh.elements._elements_in_region(ri)
            dis_u = u[start:end, :]
            dis_v = v[start:end, :]
            metric = J(range(start, end))
            diff = dis_u ** d + dis_v ** d
            integral.append(
                np.einsum(
                    'eij, i, j -> ',
                    diff * metric,
                    *quad_weights,
                    optimize='optimal'
                )
            )

        return np.sum(integral) ** (1/d)

    def _L_norm_m2_n2_k0(self, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        v = v[0]

        integral = list()
        for ri in self._mesh.regions:
            start, end = self._mesh.elements._elements_in_region(ri)
            dis_v = v[start:end, :]
            metric = J(range(start, end))

            diff = dis_v ** d
            integral.append(
                np.einsum(
                    'eij, i, j -> ',
                    diff * metric,
                    *quad_weights,
                    optimize='optimal'
                )
            )

        return np.sum(integral) ** (1/d)

    def _L_norm_m3_n3_k0(self, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        v = v[0]

        integral = list()
        for ri in self._mesh.regions:
            start, end = self._mesh.elements._elements_in_region(ri)
            dis_v = v[start:end, :]
            metric = J(range(start, end))

            diff = dis_v ** d
            integral.append(
                np.einsum(
                    'eijk, i, j, k -> ',
                    diff * metric,
                    *quad_weights,
                    optimize='optimal'
                )
            )

        return np.sum(integral) ** (1/d)

    def _L_norm_m3_n3_k1(self, d, quad_degree, local_cochain, degree):
        """"""
        return self._L_norm_m3_n3_k2(d, quad_degree, local_cochain, degree)

    def _L_norm_m3_n3_k2(self, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        u, v, w = v

        integral = list()
        for ri in self._mesh.regions:
            start, end = self._mesh.elements._elements_in_region(ri)
            dis_u = u[start:end, :]
            dis_v = v[start:end, :]
            dis_w = w[start:end, :]
            metric = J(range(start, end))
            diff = dis_u ** d + dis_v ** d + dis_w ** d
            integral.append(
                np.einsum(
                    'eijk, i, j, k -> ',
                    diff * metric,
                    *quad_weights,
                    optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _L_norm_m3_n3_k3(self, d, quad_degree, local_cochain, degree):
        """"""
        return self._L_norm_m3_n3_k0(d, quad_degree, local_cochain, degree)
