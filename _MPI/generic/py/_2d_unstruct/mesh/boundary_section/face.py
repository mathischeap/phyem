# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class _2d_Face(Frozen):
    """The element edge as a separate class, and we call it a Face."""

    def __init__(self, mesh, element_index, edge_index):
        """"""
        self._mesh = mesh
        self._element_index = element_index
        self._edge_index = edge_index
        self._freeze()

    def __repr__(self):
        """repr"""
        return f"<Face {self._element_index}:{self._edge_index} on {self._mesh}>"
