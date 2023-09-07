# -*- coding: utf-8 -*-
r"""
"""

import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class MseHyPy2SpaceErrorLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._mesh = space.mesh
        self._space = space
        self._freeze()

    def __call__(self, cf, cochain, quad_degree, d=2):
        """default: L^d norm"""
        return self.L(cf, cochain, quad_degree, d=d)

    def L(self, cf, cochain, quad_degree, d=2):
        """compute the L^d norm of the root-form"""
        abs_sp = self._space.abstract
        k = abs_sp.k
        degree = cochain._f.degree
        assert isinstance(d, int) and d > 0, f"d={d} is wrong."

        if quad_degree is None:
            quad_degree = [i + 3 for i in self._space[degree].p]  # + 3 for higher accuracy.
        else:
            pass

        return getattr(self, f'_k{k}')(d, cf, cochain, quad_degree)

    def _k2(self, d, cf, cochain, quad_degree):
        """m2 n2 k2"""
        return self._k0(d, cf, cochain, quad_degree)

    def _k1(self, d, cf, cochain, quad_degree):
        """"""
        g = cochain._g
        t = cochain._t
        representative = self._mesh[g]
        quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad
        xy, v = self._space.reconstruct(g, cochain, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = representative.ct.Jacobian(*quad_nodes)
        x, y = xy
        u, v = v
        integral = list()
        func = cf.field[t]

        for e in v:
            fc = representative[e]
            region = fc.region
            vector = func[region]
            ext_u, ext_v = vector(x[e], y[e])
            dis_u = u[e]
            dis_v = v[e]
            metric = J[e]
            diff = (dis_u - ext_u) ** d + (dis_v - ext_v) ** d
            integral.append(
                np.einsum('ij, ij, i, j -> ', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)

    def _k0(self, d, cf, cochain, quad_degree):
        """"""
        g = cochain._g
        t = cochain._t
        representative = self._mesh[g]
        quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad
        xy, v = self._space.reconstruct(g, cochain, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = representative.ct.Jacobian(*quad_nodes)
        x, y = xy
        v = v[0]
        func = cf.field[t]

        integral = list()
        for e in v:
            fc = representative[e]
            region = fc.region
            scalar = func[region]
            ext_v = scalar(x[e], y[e])[0]
            dis_v = v[e]
            metric = J[e]

            diff = (dis_v - ext_v) ** d
            integral.append(
                np.einsum('ij, ij, i, j -> ', diff, metric, *quad_weights, optimize='optimal')
            )

        return np.sum(integral) ** (1/d)
