# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MsePyT2DBoundary(Frozen):
    """"""

    def __init__(self, msepy_manifold):
        """"""
        self._manifold = msepy_manifold
        self._freeze()

    def config(self, boundary_section_mesh, vco):
        """on `boundary_section_mesh`, it values as ``vco`` (vector calculus object).

        Parameters
        ----------
        boundary_section_mesh
        vco

        Returns
        -------

        """
