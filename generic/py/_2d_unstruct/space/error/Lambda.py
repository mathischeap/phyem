# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK, MASTER_RANK, COMM, SIZE
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class ErrorLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._mesh = space.mesh
        self._space = space
        self._freeze()

    def __call__(self, cf, cochain, d=2):
        """default: L^d norm"""
        return self.L(cf, cochain, d=d)

    def L(self, cf, cochain, d=2):
        """compute the L^d norm of the root-form"""
        abs_sp = self._space.abstract
        k = abs_sp.k
        degree = cochain._f.degree
        assert isinstance(d, int) and d > 0, f"d={d} is wrong."
        p = self._space[degree].p
        quad_degree = [p + 3, p + 3]  # + 3 for higher accuracy.
        return getattr(self, f'_k{k}')(d, cf, cochain, quad_degree)

    def _k2(self, d, cf, cochain, quad_degree):
        """k2"""
        return self._k0(d, cf, cochain, quad_degree)

    def _k1(self, d, cf, cochain, quad_degree):
        """"""
        t = cochain._t
        quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad
        xy, v = self._space.reconstruct(cochain, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y = xy
        u, v = v
        integral = list()

        for e in v:
            ext_u, ext_v = cf(t, x[e], y[e])
            dis_u = u[e]
            dis_v = v[e]
            metric = J[e]
            diff = (dis_u - ext_u) ** d + (dis_v - ext_v) ** d
            integral.append(
                np.einsum(
                    'ij, ij, i, j -> ',
                    diff, metric, *quad_weights,
                    optimize='optimal')
            )

        rank_error = np.sum(integral)

        if SIZE == 1:
            return rank_error ** (1/d)

        else:
            rank_error = COMM.gather(rank_error, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                rank_error = sum(rank_error) ** (1/d)
            else:
                pass
            return COMM.bcast(rank_error, root=MASTER_RANK)

    def _k0(self, d, cf, cochain, quad_degree):
        """"""
        t = cochain._t
        quad_nodes, quad_weights = Quadrature(quad_degree, category='Gauss').quad
        xy, v = self._space.reconstruct(cochain, *quad_nodes, ravel=False)
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')
        J = self._mesh.ct.Jacobian(*quad_nodes)
        x, y = xy
        v = v[0]

        integral = list()
        for e in v:
            ext_v = cf(t, x[e], y[e])[0]
            dis_v = v[e]
            metric = J[e]

            diff = (dis_v - ext_v) ** d
            integral.append(
                np.einsum(
                    'ij, ij, i, j -> ',
                    diff, metric, *quad_weights,
                    optimize='optimal')
            )

        rank_error = np.sum(integral)

        if SIZE == 1:
            return rank_error ** (1/d)

        else:
            rank_error = COMM.gather(rank_error, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                rank_error = sum(rank_error) ** (1/d)
            else:
                pass
            return COMM.bcast(rank_error, root=MASTER_RANK)
