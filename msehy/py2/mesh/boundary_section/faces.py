# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2BoundarySectionFaces(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._freeze()

    def _renew(self):
        """"""
