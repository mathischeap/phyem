# -*- coding: utf-8 -*-
r"""
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from scipy.sparse.linalg import inv


class MsePySpaceProjection(Frozen):
    """"""

    def __init__(self, from_space, to_space, from_degree, to_degree):
        """"""
        from_mesh = from_space.mesh
        to_mesh = to_space.mesh
        assert from_mesh is to_mesh
        self._mesh = from_mesh
        self._from_space = from_space
        self._to_space = to_space
        self._from_degree = from_degree
        self._to_degree = to_degree
        self._freeze()

    @property
    def matrix(self):
        """
        Very important:

        from -> axis 1
        to -> axis 0

        Returns
        -------

        """
        M = self._to_space.mass_matrix(self._to_degree)
        W = self._to_space.inner_product(self._to_degree, (self._from_space, self._from_degree))
        dd = dict()
        for re in M:
            Mre = M[re]
            Wre = W[re]
            inv_Mre = inv(Mre.tocsc())
            Pre = inv_Mre @ Wre
            dd[re] = Pre

        gm0 = self._to_space.gathering_matrix(self._to_degree)
        gm1 = self._from_space.gathering_matrix(self._from_degree)

        dd = M.__class__(M._mp, dd)

        return MsePyStaticLocalMatrix(
            dd, gm0, gm1
        )
