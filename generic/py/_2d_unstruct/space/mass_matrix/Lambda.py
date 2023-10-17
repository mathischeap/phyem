# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix, bmat

from tools.frozen import Frozen
from tools.quadrature import Quadrature

from src.spaces.main import _degree_str_maker

from generic.py.matrix.localize.static import Localize_Static_Matrix

_global_cache_0_ = {}
_global_cache_1_inner_ = {}
_global_cache_1_outer_ = {}
_global_cache_2_ = {}


class MassMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._cache = {}
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        key = _degree_str_maker(degree)
        if key in self._cache:
            M = self._cache[key]
        else:
            k = self._k
            if k == 1:  # for k == 0 and k == 1.
                method_name = f"_k{k}_{self._orientation}"
            else:
                method_name = f"_k{k}"
            M = getattr(self, method_name)(degree)
            self._cache[key] = M  # M is the metadata, will never been touched.
        gm = self._space.gathering_matrix(degree)
        M = Localize_Static_Matrix(M, gm, gm)
        return M

    def _k0(self, degree):
        """mass matrix of 2-form on 2-manifold in 2d space."""
        p = self._space[degree].p
        quad = Quadrature([p+2, p+2], category='Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions(degree, *quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et)
        M = dict()
        for index in detJM:  # go through all reference elements
            element = self._mesh[index]
            metric_signature = element.metric_signature
            if metric_signature in _global_cache_0_:
                M[index] = _global_cache_0_[metric_signature]
            else:
                det_jm = detJM[index]
                bf = BF[index][0]
                M_re = np.einsum(
                    'im, jm, m -> ij',
                    bf, bf, det_jm * quad_weights,
                    optimize='optimal',
                            )
                m = csr_matrix(M_re)
                _global_cache_0_[metric_signature] = m
                M[index] = m
        return M

    def _k2(self, degree):
        """mass matrix of 2-form on 2-manifold in 2d space."""
        p = self._space[degree].p
        quad = Quadrature([p+2, p+2], category='Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions(degree, *quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et)
        M = dict()
        for index in detJM:  # go through all reference elements
            element = self._mesh[index]
            metric_signature = element.metric_signature
            if metric_signature in _global_cache_2_:
                M[index] = _global_cache_2_[metric_signature]
            else:
                det_jm = np.reciprocal(detJM[index])
                bf = BF[index][0]
                M_re = np.einsum(
                    'im, jm, m -> ij',
                    bf, bf, det_jm * quad_weights,
                    optimize='optimal',
                )
                m = csr_matrix(M_re)
                _global_cache_2_[metric_signature] = m
                M[index] = m

        return M

    @staticmethod
    def _einsum_helper(metric, bfO, bfS):
        """"""
        M = np.einsum('m, im, jm -> ij', metric, bfO, bfS, optimize='optimal')
        return csr_matrix(M)

    def _k1_outer(self, degree):
        """mass matrix of outer 1-form on 2-manifold in 2d space."""
        p = self._space[degree].p
        quad = Quadrature([p+2, p+2], category='Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions(degree, *quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et)
        G = self._space.mesh.ct.inverse_metric_matrix(*xi_et)
        M = dict()
        cache = {}
        csm = self._space.basis_functions.csm(degree)
        for index in detJM:
            if index in csm:
                metric_signature = index
                use_global_cache = False
            else:
                element = self._mesh[index]
                metric_signature = element.metric_signature
                use_global_cache = True

            if metric_signature in cache:
                M[index] = cache[metric_signature]
            else:
                if use_global_cache and metric_signature in _global_cache_1_outer_:
                    m = _global_cache_1_outer_[metric_signature]
                else:
                    det_jm = detJM[index]
                    g = G[index]
                    bf = BF[index]
                    M00 = self._einsum_helper(quad_weights * det_jm * g[1][1], bf[0], bf[0])
                    M11 = self._einsum_helper(quad_weights * det_jm * g[0][0], bf[1], bf[1])

                    M01 = - self._einsum_helper(quad_weights * det_jm * g[1][0], bf[0], bf[1])
                    M10 = M01.T

                    m = bmat(
                        [
                            (M00, M01),
                            (M10, M11)
                        ], format='csr'
                    )

                    if use_global_cache:
                        _global_cache_1_outer_[metric_signature] = m
                    else:
                        pass

                cache[metric_signature] = m
                M[index] = m

        return M

    def _k1_inner(self, degree):
        """mass matrix of outer 1-form on 2-manifold in 2d space."""
        p = self._space[degree].p
        quad = Quadrature([p+2, p+2], category='Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, BF = self._space.basis_functions(degree, *quad_nodes)
        detJM = self._space.mesh.ct.Jacobian(*xi_et)
        G = self._space.mesh.ct.inverse_metric_matrix(*xi_et)
        M = dict()
        cache = {}
        csm = self._space.basis_functions.csm(degree)
        for index in detJM:
            if index in csm:
                metric_signature = index
                use_global_cache = False
            else:
                element = self._mesh[index]
                metric_signature = element.metric_signature
                use_global_cache = True

            if metric_signature in cache:
                M[index] = cache[metric_signature]
            else:
                if use_global_cache and metric_signature in _global_cache_1_inner_:
                    m = _global_cache_1_inner_[metric_signature]
                else:
                    det_jm = detJM[index]
                    g = G[index]
                    bf = BF[index]
                    M00 = self._einsum_helper(quad_weights * det_jm * g[0][0], bf[0], bf[0])
                    M11 = self._einsum_helper(quad_weights * det_jm * g[1][1], bf[1], bf[1])
                    M01 = self._einsum_helper(quad_weights * det_jm * g[0][1], bf[0], bf[1])
                    M10 = M01.T

                    m = bmat(
                        [
                            (M00, M01),
                            (M10, M11)
                        ], format='csr'
                    )

                    if use_global_cache:
                        _global_cache_1_inner_[metric_signature] = m
                    else:
                        pass

                cache[metric_signature] = m
                M[index] = m

        return M
