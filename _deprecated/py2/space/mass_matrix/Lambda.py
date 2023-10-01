# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix, bmat
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from src.config import _setting


class MseHyPy2MassMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._freeze()

    def __call__(self, degree, g):
        """"""
        g = self._space._pg(g)
        is_linear = self._space.mesh.background.elements._is_linear()
        if is_linear:  # ALL elements are linear.
            high_accuracy = _setting['high_accuracy']
            if high_accuracy:
                quad_degree = [p + 1 for p in self._space[degree].p]
                # +1 for conservation
            else:
                quad_degree = [p for p in self._space[degree].p]
                # + 0 for much sparser matrices.
        else:
            quad_degree = [p + 2 for p in self._space[degree].p]

        k = self._k

        if k == 1:
            method_name = f"_k{k}_{self._orientation}"
        else:
            method_name = f"_k{k}"
        M = getattr(self, method_name)(degree, g, quad_degree)

        return M

    def _k0(self, degree, g, quad_degree):
        """mass matrix of 0-form on 2-manifold in 2d space"""
        quad = Quadrature(quad_degree, category='Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions(degree, g, *quad_nodes)
        representative = self._space.mesh[g]
        detJM = representative.ct.Jacobian(*xi_et)
        M = dict()
        cache = dict()
        for index in detJM:  # go through all fc
            fc = representative[index]
            metric_signature = fc.metric_signature

            if isinstance(metric_signature, str) and metric_signature in cache:
                M[index] = cache[metric_signature]
            else:
                bf = BF[index][0]
                det_jm = detJM[index]
                M_re = np.einsum(
                    'im, jm, m -> ij',
                    bf, bf, det_jm * quad_weights,
                    optimize='optimal',
                            )
                M_re = csr_matrix(M_re)
                if isinstance(metric_signature, str):
                    cache[metric_signature] = M_re
                else:
                    pass

                M[index] = M_re

        return M

    def _k2(self, degree, g, quad_degree):
        """mass matrix of 2-form on 2-manifold in 2d space"""
        quad = Quadrature(quad_degree, category='Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions(degree, g, *quad_nodes)
        representative = self._space.mesh[g]
        detJM = representative.ct.Jacobian(*xi_et)
        M = dict()
        cache = dict()
        for index in detJM:  # go through all fc
            # fc = representative[index]
            # metric_signature = fc.metric_signature
            #
            # if isinstance(metric_signature, str) and metric_signature in cache:
            #     M[index] = cache[metric_signature]
            # else:
            bf = BF[index][0]
            det_jm = np.reciprocal(detJM[index])

            M_re = np.einsum(
                'im, jm, m -> ij',
                bf, bf, det_jm * quad_weights,
                optimize='optimal',
                        )
            M_re = csr_matrix(M_re)
                # if isinstance(metric_signature, str):
                #     cache[metric_signature] = M_re
                # else:
                #     pass

            M[index] = M_re
        return M

    def _k1_inner(self, degree, g, quad_degree):
        """mass matrix of inner 1-form on 2-manifold in 2d space"""
        quad = Quadrature(quad_degree, category='Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions(degree, g, *quad_nodes)
        representative = self._space.mesh[g]
        detJM = representative.ct.Jacobian(*xi_et)
        G = representative.ct.inverse_metric_matrix(*xi_et)
        M = dict()
        cache = dict()
        for index in detJM:
            fc = representative[index]
            metric_signature = fc.metric_signature

            if isinstance(metric_signature, str) and metric_signature in cache:
                M[index] = cache[metric_signature]
            else:
                bf = BF[index]
                det_jm = detJM[index]
                g = G[index]
                M00 = self._einsum_helper(quad_weights * det_jm * g[0][0], bf[0], bf[0])
                M11 = self._einsum_helper(quad_weights * det_jm * g[1][1], bf[1], bf[1])

                if isinstance(metric_signature, str) and metric_signature[:6] == 'Linear':
                    M01 = None
                    M10 = None
                else:
                    M01 = self._einsum_helper(quad_weights * det_jm * g[0][1], bf[0], bf[1])
                    M10 = M01.T

                M_re = bmat(
                    [
                        (M00, M01),
                        (M10, M11)
                    ], format='csr'
                )
                if isinstance(metric_signature, str):
                    cache[metric_signature] = M_re
                else:
                    pass

                M[index] = M_re

        return M

    @staticmethod
    def _einsum_helper(metric, bfO, bfS):
        """"""
        M = np.einsum('m, im, jm -> ij', metric, bfO, bfS, optimize='optimal')
        return csr_matrix(M)

    def _k1_outer(self, degree, g, quad_degree):
        """mass matrix of outer 1-form on 2-manifold in 2d space"""
        quad = Quadrature(quad_degree, category='Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions(degree, g, *quad_nodes)
        representative = self._space.mesh[g]
        detJM = representative.ct.Jacobian(*xi_et)
        G = representative.ct.inverse_metric_matrix(*xi_et)
        M = dict()
        cache = dict()
        for index in detJM:
            # fc = representative[index]
            # metric_signature = fc.metric_signature
            #
            # if isinstance(metric_signature, str) and metric_signature in cache:
            #     M[index] = cache[metric_signature]
            # else:

            bf = BF[index]

            det_jm = detJM[index]
            g = G[index]

            M00 = self._einsum_helper(quad_weights * det_jm * g[1][1], bf[0], bf[0])
            M11 = self._einsum_helper(quad_weights * det_jm * g[0][0], bf[1], bf[1])

            # if isinstance(metric_signature, str) and metric_signature[:6] == 'Linear':
            #     M01 = None
            #     M10 = None
            # else:
            M01 = - self._einsum_helper(quad_weights * det_jm * g[1][0], bf[0], bf[1])
            M10 = M01.T

            M_re = bmat(
                [
                    (M00, M01),
                    (M10, M11)
                ], format='csr'
            )
                # if isinstance(metric_signature, str):
                #     cache[metric_signature] = M_re
                # else:
                #     pass

            M[index] = M_re

        return M
