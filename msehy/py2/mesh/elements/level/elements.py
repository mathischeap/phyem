# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.mesh.elements.level.coordinate_tranformation import MseHyPy2MeshElementsLevelElementsCT


class MseHyPy2MeshElementsLevelElements(Frozen):
    """"""

    def __init__(self, level):
        """"""
        self._level = level
        self._mesh = level._mesh
        self._level_num = level._level_num
        self._ct = MseHyPy2MeshElementsLevelElementsCT(self)
        self._freeze()

    def __repr__(self):
        """repr"""
        return rf"<G[{self._level._mesh.___generation___}] elements of level#{self._level_num} of {self._mesh}>"

    @property
    def ct(self):
        """"""
        return self._ct
