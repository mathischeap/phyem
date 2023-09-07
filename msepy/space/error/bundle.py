# -*- coding: utf-8 -*-
r"""
"""

import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class MsePySpaceErrorBundle(Frozen):
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
            quad_degrees = list()
            for _I_ in range(n):
                quad_degree = [i + 3 for i in self._space[degree].p[_I_]]  # + 3 for higher accuracy.
                quad_degrees.append(
                    quad_degree
                )
            quad_degrees = np.array(quad_degrees)
            quad_degree = np.max(quad_degrees, axis=0)

        else:
            raise NotImplementedError()

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
                np.einsum('ij, ij -> ', diff, metric * quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _m2_n2_k2(self, cf, t, d, quad_degree, local_cochain, degree):
        """m2 n2 k2"""
        return self._m2_n2_k0(cf, t, d, quad_degree, local_cochain, degree)

    def _m2_n2_k0(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        xy, V = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y = xy
        u, v = V
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
                np.einsum('eij, eij, i, j -> ', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _m2_n2_k1(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        xy, V = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y = xy
        v00, v01 = V[0]
        v10, v11 = V[1]
        integral = list()

        for ri in cf.field:
            tensor = cf.field[ri][t]  # the vector evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[start:end, :]
            y_region = y[start:end, :]
            ext_v0, ext_v1 = tensor(x_region, y_region)
            ext_v00, ext_v01 = ext_v0
            ext_v10, ext_v11 = ext_v1
            dis_v00 = v00[start:end, :]
            dis_v01 = v01[start:end, :]
            dis_v10 = v10[start:end, :]
            dis_v11 = v11[start:end, :]
            metric = J(range(start, end))

            diff = ((dis_v00 - ext_v00) ** d + (dis_v01 - ext_v01) ** d +
                    (dis_v10 - ext_v10) ** d + (dis_v11 - ext_v11) ** d)

            integral.append(
                np.einsum('eij, eij, i, j -> ', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _m3_n3_k3(self, cf, t, d, quad_degree, local_cochain, degree):
        """m2 n2 k2"""
        return self._m3_n3_k0(cf, t, d, quad_degree, local_cochain, degree)

    def _m3_n3_k0(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        xyz, V = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y, z = xyz
        u, v, w = V
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
                np.einsum('eij, eij, i, j -> ', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _m3_n3_k1(self, cf, t, d, quad_degree, local_cochain, degree):
        """m2 n2 k2"""
        return self._m3_n3_k2(cf, t, d, quad_degree, local_cochain, degree)

    def _m3_n3_k2(self, cf, t, d, quad_degree, local_cochain, degree):
        """"""
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        xyz, V = self._space.reconstruct(local_cochain, degree, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y, z = xyz
        v00, v01, v02 = V[0]
        v10, v11, v12 = V[1]
        v20, v21, v22 = V[2]
        integral = list()

        for ri in cf.field:
            tensor = cf.field[ri][t]  # the vector evaluated at time `t`.
            start, end = self._mesh.elements._elements_in_region(ri)
            x_region = x[start:end, :]
            y_region = y[start:end, :]
            z_region = z[start:end, :]
            ext_v0, ext_v1, ext_v2 = tensor(x_region, y_region, z_region)
            ext_v00, ext_v01, ext_v02 = ext_v0
            ext_v10, ext_v11, ext_v12 = ext_v1
            ext_v20, ext_v21, ext_v22 = ext_v2
            dis_v00 = v00[start:end, :]
            dis_v01 = v01[start:end, :]
            dis_v02 = v02[start:end, :]
            dis_v10 = v10[start:end, :]
            dis_v11 = v11[start:end, :]
            dis_v12 = v12[start:end, :]
            dis_v20 = v20[start:end, :]
            dis_v21 = v21[start:end, :]
            dis_v22 = v22[start:end, :]
            metric = J(range(start, end))
            diff = ((dis_v00 - ext_v00) ** d + (dis_v01 - ext_v01) ** d + (dis_v02 - ext_v02) ** d +
                    (dis_v10 - ext_v10) ** d + (dis_v11 - ext_v11) ** d + (dis_v12 - ext_v12) ** d +
                    (dis_v20 - ext_v20) ** d + (dis_v21 - ext_v21) ** d + (dis_v22 - ext_v22) ** d)
            integral.append(
                np.einsum('eij, eij, i, j -> ', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)
