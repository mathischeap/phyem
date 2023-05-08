# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from tools.quadrature import Quadrature
import numpy as np
from scipy.sparse import csr_matrix


class MsePyMassMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._m = space.abstract.m  # dimensions of the embedding space.
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._freeze()

    def __call__(self, degree, quad_degree=None):
        """Making the local numbering for degree."""
        key = f"{degree}" + str(quad_degree)

        if key in self._cache:
            M = self._cache[key]
        else:
            if self._n == 2:  # for k == 0 and k == 1.
                method_name = f"_m{self._m}_n{self._n}_k{self._k}_{self._orientation}"
            else:
                method_name = f"_m{self._m}_n{self._n}_k{self._k}"
            M = getattr(self, method_name)(degree, quad_degree=quad_degree)
            self._cache[key] = M

        return M

    def _m3_n3_k3(self, degree, quad_degree=None):
        """mass matrix of 3-form on 3-manifold in 3d space"""

    def _m3_n3_k2(self, degree, quad_degree=None):
        """mass matrix of 2-form on 3-manifold in 3d space"""

    def _m3_n3_k1(self, degree, quad_degree=None):
        """mass matrix of 1-form on 3-manifold in 3d space"""

    def _m3_n3_k0(self, degree, quad_degree=None):
        """mass matrix of 0-form on 3-manifold in 3d space"""

    def _m2_n2_k0(self, degree, quad_degree=None):
        """mass matrix of 0-form on 2-manifold in 2d space"""

    def _m2_n2_k2(self, degree, quad_degree=None):
        """mass matrix of 2-form on 2-manifold in 2d space"""
        if quad_degree is None:
            quad_degree = [p + 2 for p in self._space[degree].p]
        else:
            pass
        quad_nodes, quad_weights = Quadrature(quad_degree).quad
        xi_et, bf = self._space.basis_functions[degree](*quad_nodes)


    def _m2_n2_k1_inner(self, degree, quad_degree=None):
        """mass matrix of inner 1-form on 2-manifold in 2d space"""

    def _m2_n2_k1_outer(self, degree, quad_degree=None):
        """mass matrix of outer 1-form on 2-manifold in 2d space"""

    def _m1_n1_k0(self, degree, quad_degree=None):
        """mass matrix of 0-form on 1-manifold in 1d space"""

    def _m1_n1_k1(self, degree, quad_degree=None):
        """mass matrix of 1-form on 1-manifold in 1d space"""
