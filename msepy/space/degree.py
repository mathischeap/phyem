# -*- coding: utf-8 -*-
"""
How to parse a degree for Mse Py space.

@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from msepy.tools.polynomials import _OneDimPolynomial


class MsePySpaceDegree(Frozen):
    """"""

    def __init__(self, space, degree):
        """"""
        self._space = space
        self._degree = degree
        self._p = None
        self._nodes = None
        self._edges = None
        self._bfs = None
        self._freeze()

    @property
    def p(self):
        """(px, py, ...) of all elements."""
        if self._p is None:
            if isinstance(self._degree, (int, float)):
                # degree = 3
                assert self._degree % 1 == 0 and self._degree > 0, f"degree wrong."
                p = tuple([self._degree for _ in range(self._space.n)])

            elif isinstance(self._degree, (list, tuple)) and all([isinstance(_, int) for _ in self._degree]):
                # degree = (3, 2, ...)
                assert len(self._degree) == self._space.n, f"degree dimension wrong."
                p = tuple(self._degree)

            else:
                raise NotImplementedError()

            self._p = p

        return self._p

    @property
    def nodes(self):
        """nodes"""
        if self._nodes is None:
            if isinstance(self._degree, int):
                # degree = 3
                assert self._degree % 1 == 0 and self._degree > 0, f"degree wrong."
                self._nodes = tuple(
                    [Quadrature(self._degree, category='Lobatto').quad[0] for _ in range(self._space.n)]
                )

            elif isinstance(self._degree, (list, tuple)) and all([isinstance(_, int) for _ in self._degree]):
                # degree = (3, 2, ...)
                assert len(self._degree) == self._space.n, f"degree dimension wrong."
                self._nodes = tuple(
                    [Quadrature(_, category='Lobatto').quad[0] for _ in self.p]
                )

            else:
                raise NotImplementedError()

        return self._nodes

    @property
    def edges(self):
        """edges"""
        if self._edges is None:
            self._edges = tuple([nodes[1:]-nodes[:-1] for nodes in self._nodes])
        return self._edges

    @property
    def bfs(self):
        """1d basis functions."""
        if self._bfs is None:
            self._bfs = tuple([_OneDimPolynomial(nodes) for nodes in self.nodes])
        return self._bfs
