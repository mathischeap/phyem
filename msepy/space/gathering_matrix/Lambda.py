# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
import numpy as np


class MsePyGatheringMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p
        return getattr(self, f"_n{self._n}_k{self._k}")(p)

    @staticmethod
    def _n3_k3(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n3_k2(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n3_k1(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n3_k0(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n2_k0(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n2_k1(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n2_k2(p):
        """"""
        raise NotImplementedError

    def _n1_k0(self, p):
        """"""
        element_map = self._mesh.elements.map
        gm = - np.ones((self._mesh.elements._num, self._space.num_local_dofs.Lambda._n1_k0(p)), dtype=int)
        current = 0
        p = p[0]
        for e, mp in enumerate(element_map):
            # number x- node
            x_m = mp[0]
            if x_m == -1 or x_m > e:  # x- side of element #e is a boundary or not numbered
                gm[e, 0] = current
                current += 1
            else:
                gm[e, 0] = gm[x_m, -1]
            # node intermediate nodes
            gm[e, 1:-1] = np.arange(current, current + p - 1)
            current += p - 1

            # number x+ node
            x_p = mp[-1]
            if x_p == -1 or x_p > e:
                gm[e, -1] = current
                current += 1
            else:
                gm[e, -1] = gm[x_p, 0]
        return gm

    @staticmethod
    def _n1_k1(p):
        """"""
        raise NotImplementedError
