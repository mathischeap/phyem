# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.src.config import RANK, MASTER_RANK
from phyem.msehtt.static.manifold.predefined.chaotic import ___invA___


def quad(A=(0, 0), B=(1, 0), C=(2, 1), D=(1, 1), periodic=False):
    r"""

    The numbering of region nodes in the reference region.

          ^ s
          |
          |
          |  node 3        face 2         node 2
          |     -----------------------------
          |     |                           |
          |     |                           |
          |     |                           |
          |     | face 3                    | face 1
          |     |                           |
          |     |                           |
          |     |                           |
          |     |                           |
          |     -----------------------------
          |   node 0         face 0       node 1
          |
          |
          ------------------------------------------> r

        """
    assert RANK == MASTER_RANK, f"only initialize quad mesh in the master rank"

    REGIONS = {
        0: _MAP_(A, B, C, D)
    }

    region_map = None        # the config method will parse the region map.
    if periodic:
        raise Exception(f'Quad region cannot be periodic. Use chaotic.')
    else:
        periodic_setting = None

    return REGIONS, region_map, periodic_setting


class _MAP_(Frozen):
    r""""""
    def __init__(self, A, B, C, D):
        r""""""
        x = np.array([A[0], B[0], C[0], D[0]])
        y = np.array([A[1], B[1], C[1], D[1]])

        alpha = ___invA___ @ x
        beta = ___invA___ @ y

        self._a1, self._a2, self._a3, self._a4 = alpha
        self._b1, self._b2, self._b3, self._b4 = beta

        self._freeze()

    @property
    def ndim(self):
        r"""This is a 2d region."""
        return 2

    @property
    def etype(self):
        r"""The element made in this region can only be of this type."""
        return 9

    def mapping(self, r, s):
        r""""""
        q = self._a1 + self._a2 * r + self._a3 * s + self._a4 * r * s
        w = self._b1 + self._b2 * r + self._b3 * s + self._b4 * r * s
        return q, w

    # noinspection PyUnusedLocal
    def Jacobian_matrix(self, r, s):
        r""""""
        raise NotImplementedError()
