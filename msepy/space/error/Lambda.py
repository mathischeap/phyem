# -*- coding: utf-8 -*-
r"""
"""

import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class MsePySpaceErrorLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._mesh = space.mesh
        self._space = space
        self._freeze()

    def __call__(self, cf, t, local_cochain, degree, quad_degree, d=2):
        """default: L^d norm"""
        return self.L(cf, t, local_cochain, degree, quad_degree, d=d)

    def L(self, cf, t, local_cochain, degree, quad_degree, d=2):
        """compute the L^d norm of the root-form"""
        abs_sp = self._space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        assert isinstance(d, int) and d > 0, f"d={d} is wrong."

        if quad_degree is None:
            quad_degree = [i + 3 for i in self._space[degree].p]  # + 3 for higher accuracy.
        else:
            pass

        return getattr(self, f'_m{m}_n{n}_k{k}')(cf, t, d, quad_degree, local_cochain, degree)

    def _m1_n1_k0(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        return self._m1_n1_k1(cf, t, d, quad_degree, local_cochain, degree)

    def _m1_n1_k1(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        x, v = self._space.reconstruct(local_cochain, degree, quad_nodes)
        J = self._mesh.ct.Jacobian(quad_nodes)
        x = x[0]
        v = v[0]

        integral = list()
        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[start:end, :]
            ext_v = scalar(x_region)[0]
            dis_v = v[start:end, :]
            metric = J(range(start, end))

            diff = (dis_v - ext_v) ** d

            integral.append(
                np.einsum(
                    'ij, ij -> ',
                    diff, metric * quad_weights,
                    optimize='optimal'
                )
            )

        return np.sum(integral) ** (1/d)

    def _m2_n2_k2(self, cf, t, d, quad_degree, local_cochain, degree):
        """m2 n2 k2"""
        return self._m2_n2_k0(cf, t, d, quad_degree, local_cochain, degree)

    def _m2_n2_k1(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        xy, v = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y = xy
        u, v = v
        integral = list()

        for ri in cf.field:
            vector = cf.field[ri][t]  # the vector evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[start:end, :]
            y_region = y[start:end, :]
            ext_u, ext_v = vector(x_region, y_region)
            dis_u = u[start:end, :]
            dis_v = v[start:end, :]
            metric = J(range(start, end))
            diff = (dis_u - ext_u) ** d + (dis_v - ext_v) ** d
            integral.append(
                np.einsum(
                    'eij, eij, i, j -> ',
                    diff, metric, *quad_weights,
                    optimize='optimal'
                )
            )

        return np.sum(integral) ** (1/d)

    def _m2_n2_k0(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        xyz, v = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y = xyz
        v = v[0]

        integral = list()
        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[start:end, :]
            y_region = y[start:end, :]
            ext_v = scalar(x_region, y_region)[0]
            dis_v = v[start:end, :]
            metric = J(range(start, end))

            diff = (dis_v - ext_v) ** d
            integral.append(
                np.einsum(
                    'eij, eij, i, j -> ',
                    diff, metric, *quad_weights,
                    optimize='optimal'
                )
            )

        return np.sum(integral) ** (1/d)

    def _m3_n3_k0(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        xyz, v = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y, z = xyz
        v = v[0]

        integral = list()
        for ri in cf.field:
            scalar = cf.field[ri][t]  # the scalar evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[start:end, :]
            y_region = y[start:end, :]
            z_region = z[start:end, :]
            ext_v = scalar(x_region, y_region, z_region)[0]
            dis_v = v[start:end, :]
            metric = J(range(start, end))

            diff = (dis_v - ext_v) ** d
            integral.append(
                np.einsum(
                    'eijk, eijk, i, j, k -> ',
                    diff, metric, *quad_weights,
                    optimize='optimal'
                )
            )

        return np.sum(integral) ** (1/d)

    def _m3_n3_k1(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        return self._m3_n3_k2(cf, t, d, quad_degree, local_cochain, degree)

    def _m3_n3_k2(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        xyz, v = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y, z = xyz
        u, v, w = v

        integral = list()
        for ri in cf.field:
            vector = cf.field[ri][t]  # the vector evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[start:end, :]
            y_region = y[start:end, :]
            z_region = z[start:end, :]
            ext_u, ext_v, ext_w = vector(x_region, y_region, z_region)
            dis_u = u[start:end, :]
            dis_v = v[start:end, :]
            dis_w = w[start:end, :]
            metric = J(range(start, end))
            diff = (dis_u - ext_u) ** d + (dis_v - ext_v) ** d + (dis_w - ext_w) ** d
            integral.append(
                np.einsum(
                    'eijk, eijk, i, j, k -> ',
                    diff, metric, *quad_weights,
                    optimize='optimal'
                )
            )

        return np.sum(integral) ** (1/d)

    def _m3_n3_k3(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        return self._m3_n3_k0(cf, t, d, quad_degree, local_cochain, degree)
