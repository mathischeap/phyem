# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
from tools.frozen import Frozen
from msehy.py2.mesh.elements.level.triangles import MseHyPy2LevelTriangles


class MseHyPy2MeshLevel(Frozen):
    """"""

    def __init__(self, elements, level_num, base_elements, refining_elements):
        """"""
        # the fundamental msehy-py2 elements instance
        self._elements = elements
        self._background = elements.background

        # I am (level_num+1)th level.
        assert isinstance(level_num, int) and level_num >= 0, f'Must be.'
        self._level_num = level_num

        if level_num == 0:
            refining_elements.sort()
        else:
            refining_elements.sort(
                key=lambda x: int(x.split('=')[0])
            )

        # the basement level.
        self._base_level_elements = base_elements  # When level_num == 0, it is the background msepy mesh elements.
        self._refining_elements = refining_elements  # labels of elements of previous level on which I am refining.

        if self.num == 0:
            assert self._base_level_elements is self.background.elements, f"must be"
        else:
            assert self._base_level_elements.__class__ is MseHyPy2LevelTriangles, f'Must be.'

        # Below are my person stuffs.
        self._triangles = MseHyPy2LevelTriangles(self)
        self._freeze()

    @property
    def generation(self):
        return self._elements.generation

    @property
    def background(self):
        return self._background

    @property
    def num(self):
        """I am (num+1)th level."""
        return self._level_num

    def __repr__(self):
        """repr"""
        return rf"<G[{self.generation}] levels[{self._level_num}] of {self.background}>"
    
    @property
    def triangles(self):
        """All the elements on this level."""
        return self._triangles

    @property
    def threshold(self):
        """"""
        return self._elements.thresholds[self.num]

    def _visualize(self, fig, density, color='k'):
        """"""
        ct = self.triangles.ct
        xi = np.linspace(-1, 1, density)
        et = np.ones(density)
        xy0 = ct.mapping(xi, -et)
        xy1 = ct.mapping(xi, et)
        for t in xy0:
            x, y = xy0[t]
            plt.plot(x, y, linewidth='0.75', color=color)
            x, y = xy1[t]
            plt.plot(x, y, linewidth='0.75', color=color)
        return fig
