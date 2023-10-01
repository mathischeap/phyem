# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from msepy.tools.polynomials import _OneDimPolynomial


class Py2SpaceDegree(Frozen):
    """"""

    def __init__(self, space, degree):
        """"""
        self._space = space
        assert space.n == 2, f'must be in 2-d.'
        self._indicator = self._space.abstract.indicator
        self._parse_degree(degree)
        self._edges = None
        self._bfs = None
        self._freeze()

    def _parse_degree(self, degree):
        """We get `p`, `nodes` and `ntype` (node types). """
        self._degree = degree
        if isinstance(degree, (int, float)):
            # for example, degree = 3
            assert degree % 1 == 0 and degree > 0, f"degree wrong."
            p = int(degree)
            nodes = tuple([Quadrature(p, category='Lobatto').quad[0] for _ in range(2)])
            ntype = 'Lobatto'
        else:
            raise NotImplementedError(f"cannot parse degree={degree}.")
        self._p = p
        self._nodes = nodes
        self._ntype = ntype

    def __repr__(self):
        """repr"""
        return rf"<Degree [{self._degree}] copy of {self._space}>"

    @property
    def p(self):
        """(px, py, ...) of all elements."""
        return self._p

    @property
    def nodes(self):
        """nodes"""
        return self._nodes

    @property
    def ntype(self):
        """node type."""
        return self._ntype

    @property
    def edges(self):
        """edges"""
        if self._edges is None:
            if self._indicator in ('Lambda', ):
                self._edges = tuple([nodes[1:]-nodes[:-1] for nodes in self._nodes])
            else:
                raise NotImplementedError(self._indicator)
        return self._edges

    @property
    def bfs(self):
        """1d basis functions."""
        if self._bfs is None:
            if self._indicator in ('Lambda', ):
                self._bfs = tuple([_OneDimPolynomial(nodes) for nodes in self.nodes])
            else:
                raise NotImplementedError(self._indicator)
        return self._bfs
