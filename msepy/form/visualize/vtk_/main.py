

from tools.frozen import Frozen


class MsePyRootFormVisualizeVTK(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._t = None
        self._freeze()

    def __getitem__(self, t):
        """"""
        self._t = t
        return self