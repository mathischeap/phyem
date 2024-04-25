# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehtt.static.mesh.partial.elements.visualize.matplot import MseHttElementsPartialMeshVisualizeMatplot


class MseHttElementsPartialMeshVisualize(Frozen):
    """"""

    def __init__(self, elements):
        """"""
        self._elements = elements
        self._matplot = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        if self._matplot is None:
            self._matplot = MseHttElementsPartialMeshVisualizeMatplot(self._elements)
        return self._matplot
