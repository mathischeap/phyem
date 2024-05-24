# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehtt.static.mesh.great.visualize.matplot import MseHttGreatMeshVisualizeMatplot


class MseHttGreatMeshVisualize(Frozen):
    """"""

    def __init__(self, tgm):
        """"""
        self._tgm = tgm
        self._matplot = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        r"""Use the default visualizer to visualize the great mesh."""
        return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        r""""""
        if self._matplot is None:
            self._matplot = MseHttGreatMeshVisualizeMatplot(self._tgm)
        return self._matplot

    def _generate_element_outline_data(self, ddf=1):
        r""""""
        outline_data = {}
        for i in self._tgm.elements:
            element = self._tgm.elements[i]
            outline_data[i] = element._generate_outline_data(ddf=ddf)
        return outline_data
