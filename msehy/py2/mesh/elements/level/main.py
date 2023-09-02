# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.mesh.elements.level.elements import MseHyPy2MeshElementsLevelElements


class MseHyPy2MeshElementsLevel(Frozen):
    """"""

    def __init__(self, msehy_elements, level_num, base_elements, refining_elements):
        """"""
        # the fundamental msehy-py2 elements instance
        self._msehy_elements = msehy_elements
        self._mesh = msehy_elements.mesh

        # I am (level_num+1)th level.
        assert isinstance(level_num, int) and level_num >= 0, f'Must be.'
        self._level_num = level_num

        # the basement level.
        self._base_level_elements = base_elements  # When level_num == 0, it is the background msepy mesh elements.
        self._refining_elements = refining_elements  # labels of elements of previous level on which I am refining.

        if self.num == 0:
            assert self._base_level_elements is self._mesh.background.elements, f"must be"
        else:
            assert self._base_level_elements.__class__ is self.__class__, f'Must be.'

        # Below are my person stuffs.
        self._elements = MseHyPy2MeshElementsLevelElements(self)
        self._freeze()

    @property
    def num(self):
        """I am (num+1)th level."""
        return self._level_num

    def __repr__(self):
        """repr"""
        return rf"<G[{self._msehy_elements.generation}] levels[{self._level_num}] of {self._mesh}>"
    
    @property
    def elements(self):
        """All the elements on this level."""
        return self._elements

    @property
    def threshold(self):
        """"""
        return self._msehy_elements.thresholds[self.num]
