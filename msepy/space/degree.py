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
from msepy.tools.polynomials import _1dPolynomial


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
                assert self._degree % 1 == 0 and self._degree > 0, f"degree wrong."
                p = [self._degree for _ in range(self._space.n)]

            else:
                raise NotImplementedError()

            self._p = p

        return self._p

    @property
    def nodes(self):
        """nodes"""
        if self._nodes is None:
            if isinstance(self._degree, int):
                nodes = Quadrature(self._degree, category='Lobatto').quad[0]
                nodes = [nodes for _ in range(self._space.n)]
                self._nodes = nodes
            else:
                raise NotImplementedError()

        return self._nodes

    @property
    def edges(self):
        """edges"""
        if self._edges is None:
            self._edges = list()
            for nodes in self._nodes:
                self._edges.append(nodes[1:]-nodes[:-1])
        return self._edges

    @property
    def bfs(self):
        """1d basis functions."""
        if self._bfs is None:
            self._bfs = list()
            for nodes in self.nodes:
                self._bfs.append(_1dPolynomial(nodes))
        return self._bfs
