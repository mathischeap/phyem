# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msehy.py2.form.visualize.matplot import MseHyPy2RootFormVisualizeMatplot
from msehy.py2.form.visualize.vtk_ import MseHyPy2RootFormVisualizeVTK


class MseHyPy2RootFormVisualize(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._t = None
        self._g = None
        self._matplot = None
        self._vtk = None
        self._quick = None
        self._freeze()

    def quick(self, density=100, **kwargs):
        r = s = np.linspace(0, 1, density)
        r, s = np.meshgrid(r, s, indexing='ij')
        dds = self._f[(self._t, self._g)].numeric.region_wise_reconstruct(r, s)
        dds.visualize(**kwargs)

    def __getitem__(self, t_g):
        """"""
        t, g = t_g
        t = self._f._pt(t)
        g = self._f._pg(g)
        self._t, self._g = t, g
        return self

    def __call__(self, *args, **kwargs):
        if 'saveto' in kwargs and kwargs['saveto'][-4:] == '.vtk':
            # when we save to .vtk file, we call the vtk visualizer.
            _vkt_kwargs = dict()
            for kw in kwargs:
                if kw != 'saveto':
                    _vkt_kwargs[kw] = kwargs[kw]
                else:
                    pass
            path = kwargs['saveto'][:-4]
            assert 'data_only' not in _vkt_kwargs and 'builder' not in _vkt_kwargs, \
                f"'data_only' and 'builder' are for self configuration, pls do not provide them"
            return self.vtk(*args, file_path=path, **_vkt_kwargs)
        else:
            pass

        return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        if self._matplot is None:
            self._matplot = MseHyPy2RootFormVisualizeMatplot(self._f)
        return self._matplot

    @property
    def vtk(self):
        if self._vtk is None:
            self._vtk = MseHyPy2RootFormVisualizeVTK(self._f)
        return self._vtk
