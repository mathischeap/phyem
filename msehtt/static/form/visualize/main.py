# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.form.visualize.quick.main import MseHttFormVisualizeQuick
from tools.vtk_.msehtt_form_static_copy import ___ph_vtk_msehtt_static_copy___


class MseHttFormVisualize(Frozen):
    r"""The visualizer of form ``f`` at time ``t``."""

    def __init__(self, f, t):
        r""""""
        self._f = f
        self._t = t
        self._quick = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        r""""""
        space = self._f.space
        m = space.m
        n = space.n
        if m == 2 and n == 2:
            return self.matplot(*args, **kwargs)
        elif m == 3 and n == 3:
            self.vtk(*args, **kwargs)
        else:
            raise NotImplementedError()

    @property
    def quick(self):
        r""""""
        if self._quick is None:
            self._quick = MseHttFormVisualizeQuick(self._f, self._t)
        return self._quick

    def vtk(self, filename, ddf=1):
        r""""""
        ___ph_vtk_msehtt_static_copy___(filename, self._f[self._t], ddf=ddf)

    def matplot(self, ddf=1, **kwargs):
        r"""We use the matplot of dds-rws to do this."""
        rws = self._f.numeric.rws(self._t, ddf=ddf)
        if rws is None:
            pass
        else:
            rws.visualize(**kwargs)
