# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2MeshElementsLevelElementsTriangle(Frozen):
    """"""

    def __init__(self, level_elements, index):
        self._elements = level_elements
        self._level = level_elements._level
        self._index = index
        self._freeze()

    def __repr__(self):
        """"""
        return f"<Triangle {self._index} on level[{self._level._level_num}] of {self._level._mesh}>"
    
    @property
    def pair(self):
        """"""
        return 
