

from tools.frozen import Frozen
from msepy.form.visualize.matplot.main import MsePyRootFormVisualizeMatplot
from msepy.form.visualize.vtk_.main import MsePyRootFormVisualizeVTK


class MsePyRootFormVisualize(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._matplot = None
        self._vtk = None
        self._freeze()

    def __getitem__(self, t):
        """"""
        return self.matplot[t]

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
