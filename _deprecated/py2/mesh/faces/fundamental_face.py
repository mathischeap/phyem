# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2MeshFundamentalFace(Frozen):
    """"""

    def __init__(self, elements, index):
        """"""
        self._elements = elements
        self._i = index
        self._ct = None
        self._freeze()

    @property
    def index(self):
        """
        (i, m, n) : representing a face of quadrilateral cell.
        (i, e) : representing an edge of triangle cell.
        Returns
        -------

        """
        return self._i

    def __repr__(self):
        """repr"""
        return (rf"<G[{self._elements.generation}] msehy2-boundary-face "
                rf"{self.index} UPON {self._elements.background}>")

    @property
    def ct(self):
        """"""
        if self._ct is None:
            if len(self.index) == 3:
                i, m, n = self.index
                self._ct = self._elements.background.elements[i].ct.face(m, n)
            else:
                i, e = self.index
                self._ct = self._elements[i].representative.ct.edge(e)
        return self._ct
