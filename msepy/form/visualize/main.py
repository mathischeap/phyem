# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.form.visualize.matplot import MsePyRootFormVisualizeMatplot
from msepy.form.visualize.vtk_ import MsePyRootFormVisualizeVTK


class MsePyRootFormVisualize(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._t = None
        self._matplot = None
        self._vtk = None
        self._freeze()

    def __getitem__(self, t):
        """"""
        self._t = t
        return self

    def __call__(self, *args, **kwargs):
        n = self._f.space.n
        if n == 3:
            assert 'data_only' not in kwargs and 'builder' not in kwargs, \
                f"'data_only' and 'builder' are for self configuration, pls do not provide them"
            return self.vtk(*args, **kwargs)

        else:
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
            self._matplot = MsePyRootFormVisualizeMatplot(self._f)
        return self._matplot

    @property
    def vtk(self):
        if self._vtk is None:
            self._vtk = MsePyRootFormVisualizeVTK(self._f)
        return self._vtk
