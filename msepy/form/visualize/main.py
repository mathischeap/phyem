

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
            return self.vtk(*args, **kwargs)
        else:
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
