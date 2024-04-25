# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehtt.static.mesh.partial.boundary_section.visualize.matplot import (
    MseHttBoundarySectionPartialMeshVisualizeMatplot)


class MseHttBoundarySectionPartialMeshVisualize(Frozen):
    """"""

    def __init__(self, boundary_section):
        """"""
        self._boundary_section = boundary_section
        self._matplot = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        if self._matplot is None:
            self._matplot = MseHttBoundarySectionPartialMeshVisualizeMatplot(self._boundary_section)
        return self._matplot
