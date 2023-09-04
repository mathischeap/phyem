# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2MeshFaces(Frozen):
    """"""

    def __init__(self, msepy_mesh, current_elements):
        """"""
        self._background = msepy_mesh
        self._elements = current_elements
        self._freeze()

    @property
    def background(self):
        """"""
        return self._background
