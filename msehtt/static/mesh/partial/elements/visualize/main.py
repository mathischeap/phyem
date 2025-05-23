# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.mesh.partial.elements.visualize.matplot import MseHttElementsPartialMeshVisualizeMatplot
from msehtt.static.mesh.partial.elements.visualize.vtk_ import ___vtk_m3n3_partial_mesh_elements___


class MseHttElementsPartialMeshVisualize(Frozen):
    r""""""

    def __init__(self, elements):
        r""""""
        self._elements = elements
        self._matplot = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        r""""""
        mn = self._elements.mn
        if mn == (3, 3):
            return ___vtk_m3n3_partial_mesh_elements___(self._elements, *args, **kwargs)
        else:
            return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        r""""""
        if self._matplot is None:
            self._matplot = MseHttElementsPartialMeshVisualizeMatplot(self._elements)
        return self._matplot
