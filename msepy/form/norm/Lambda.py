# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:18 PM on 5/1/2023
"""

import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class MsePyRootFormNormLambda(Frozen):
    """"""

    def __init__(self, rf, t):
        """"""
        self._f = rf
        self._t = t
        self._mesh = rf.mesh
        self._space = rf.space
        self._freeze()

    def __call__(self, *args, **kwargs):
        """default: L^d norm"""
        return self.L(*args, **kwargs)

    def L(self, d=2, quad_degree=None):
        """compute the L^d norm of the root-form"""
        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        assert isinstance(d, int) and d > 0, f"d={d} is wrong."

        if quad_degree is None:
            quad_degree = [i + 3 for i in self._space[self._f.degree].p]  # + 3 for higher accuracy.
        else:
            pass

        return getattr(self, f'_m{m}_n{n}_k{k}')(d, quad_degree)

    def _m1_n1_k0(self, d, quad_degree):
        """"""
        return self._m1_n1_k1(d, quad_degree)

    def _m1_n1_k1(self, d, quad_degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._f[self._t].reconstruct(quad_nodes)
        J = self._mesh.ct.Jacobian(quad_nodes)
        v = v[0]

        integral = list()
        for ri in self._mesh.regions:
            start, end = self._mesh.elements._elements_in_region(ri)
            dis_v = v[start:end, :]
            metric = J(range(start, end))
            integral.extend(
                np.einsum('ij, ij -> i', dis_v ** d, metric * quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _m2_n2_k2(self, *args, **kwargs):
        """m2 n2 k2"""
        return self._m2_n2_k0(*args, **kwargs)

    def _m2_n2_k1(self, d, quad_degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._f[self._t].reconstruct(*quad_nodes)
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
            integral.extend(
                np.einsum('eij, eij, i, j -> i', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _m2_n2_k0(self, d, quad_degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._f[self._t].reconstruct(*quad_nodes)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        v = v[0]

        integral = list()
        for ri in self._mesh.regions:
            start, end = self._mesh.elements._elements_in_region(ri)
            dis_v = v[start:end, :]
            metric = J(range(start, end))

            diff = dis_v ** d
            integral.extend(
                np.einsum('eij, eij, i, j -> i', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _m3_n3_k0(self, d, quad_degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._f[self._t].reconstruct(*quad_nodes)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        v = v[0]

        integral = list()
        for ri in self._mesh.regions:
            start, end = self._mesh.elements._elements_in_region(ri)
            dis_v = v[start:end, :]
            metric = J(range(start, end))

            diff = dis_v ** d
            integral.extend(
                np.einsum('eijk, eijk, i, j, k -> i', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _m3_n3_k1(self, d, quad_degree):
        """"""
        return self._m3_n3_k2(d, quad_degree)

    def _m3_n3_k2(self, d, quad_degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        _, v = self._f[self._t].reconstruct(*quad_nodes)
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
            integral.extend(
                np.einsum('eijk, eijk, i, j, k -> i', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _m3_n3_k3(self, d, quad_degree):
        """"""
        return self._m3_n3_k0(d, quad_degree)
