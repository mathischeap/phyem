# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehtt.static.form.visualize.vtk_.main import MseHttFormVisualizeVtk
from msehtt.static.form.visualize.quick.main import MseHttFormVisualizeQuick


class MseHttFormVisualize(Frozen):
    """The visualizer of form ``f`` at time ``t``."""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._vtk = None
        self._quick = None
        self._freeze()

    def __call__(self, saveto, ddf=1):
        """"""
        return self.vtk_(saveto, ddf=ddf)

    @property
    def vtk_(self):
        if self._vtk is None:
            self._vtk = MseHttFormVisualizeVtk(self._f, self._t)
        return self._vtk

    @property
    def quick(self):
        if self._quick is None:
            self._quick = MseHttFormVisualizeQuick(self._f, self._t)
        return self._quick
