# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from phyem.tools.frozen import Frozen
from phyem.tools.quadrature import Quadrature


class MsePyWedgeMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._m = space.abstract.m  # dimensions of the embedding space.
        self._orientation = space.abstract.orientation
        self._freeze()

    def __call__(self, other_space, self_degree, other_degree, quad=None):
        """"""
        if quad is None:
            quad_degree = np.max((
                [p + 1 for p in self._space[self_degree].p],
                [p + 1 for p in self._space[other_degree].p]
            ), axis=0)
            quad = (quad_degree, 'Gauss')

        else:
            raise NotImplementedError()

        m = self._m
        n = self._n
        k = self._k

        if m == 2 and n == 2 and k == 1:  # for k == 0 and k == 1.
            method_name = f"_m{m}_n{n}_k{k}_{self._orientation}"
        else:
            method_name = f"_m{m}_n{n}_k{k}"
        W = getattr(self, method_name)(other_space, self_degree, other_degree, quad)

        return W

    def _m2_n2_k0(self, other_space, self_degree, other_degree, quad):
        """wedge matrix of 0-form on 2-manifold in 2d space: 0 ^ 2.

        Self is on axis-0.

        """

        s0 = self._space
        s1 = other_space
        d0 = self_degree
        d1 = other_degree

        quad_degree, quad_type = quad
        quad = Quadrature(quad_degree, category=quad_type)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf0 = s0.basis_functions[d0](*quad_nodes)
        xi_et, bf1 = s1.basis_functions[d1](*quad_nodes)

        bf0 = bf0[0]
        bf1 = bf1[0]

        W = np.einsum(
            'im, jm, m -> ij',
            bf0, bf1, quad_weights,
            optimize='optimal',
        )
        return csr_matrix(W)
