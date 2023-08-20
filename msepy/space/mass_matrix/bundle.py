# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from tools.quadrature import Quadrature
import numpy as np
from scipy.sparse import csr_matrix, bmat
from src.config import _setting


class MsePyMassMatrixBundle(Frozen):
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

    def __call__(self, degree, quad=None):
        """Making the local numbering for degree."""
        key = f"{degree}" + str(quad)

        if quad is None:
            is_linear = self._space.mesh.elements._is_linear()
            if is_linear:  # ALL elements are linear.
                high_accuracy = _setting['high_accuracy']
                if high_accuracy:
                    quad_degree = [p + 1 for p in self._space[degree].p]
                    # +1 for conservation
                else:
                    quad_degree = [p for p in self._space[degree].p]
                    # + 0 for much sparser matrices.
                quad_type = self._space[degree].ntype
                quad = (quad_degree, quad_type)
            else:
                quad_degree = [p + 2 for p in self._space[degree].p]
                quad = (quad_degree, 'Gauss')

        else:
            pass

        m = self._m
        n = self._n
        k = self._k

        if key in self._cache:
            M = self._cache[key]
        else:
            if m == 2 and n == 2 and k == 1:  # for k == 0 and k == 1.
                method_name = f"_m{m}_n{n}_k{k}_{self._orientation}"
            else:
                method_name = f"_m{m}_n{n}_k{k}"
            M = getattr(self, method_name)(degree, quad)
            M = self._space.mesh.elements._index_mapping.distribute_according_to_reference_elements_dict(M)
            self._cache[key] = M

        return M

    def _m3_n3_k3(self, degree, quad):
        """"""
        