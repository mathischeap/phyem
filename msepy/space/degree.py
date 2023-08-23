# -*- coding: utf-8 -*-
r"""
How to parse a degree for Mse Py space.
"""
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from msepy.tools.polynomials import _OneDimPolynomial


class MsePySpaceDegree(Frozen):
    """"""

    def __init__(self, space, degree):
        """"""
        self._space = space
        self._indicator = self._space.abstract.indicator
        self._parse_degree(degree)
        self._edges = None
        self._bfs = None
        self._freeze()

    def _parse_degree(self, degree):
        """We get `p`, `nodes` and `ntype` (node types). """
        self._degree = degree

        if self._indicator in ('Lambda', 'bundle-diagonal'):
            if isinstance(degree, (int, float)):
                # for example, degree = 3
                assert degree % 1 == 0 and degree > 0, f"degree wrong."
                p = tuple([degree for _ in range(self._space.n)])
                nodes = tuple(
                    [Quadrature(degree, category='Lobatto').quad[0] for _ in range(self._space.n)]
                )
                ntype = ['Lobatto' for _ in p]
                p_shape = (self._space.n, )

            elif isinstance(degree, (list, tuple)) and all([isinstance(_, int) for _ in degree]):
                # for example, degree = (3, 2, ...)
                assert len(degree) == self._space.n, f"degree dimension wrong."
                p = tuple(self._degree)
                nodes = tuple(
                    [Quadrature(_, category='Lobatto').quad[0] for _ in p]
                )
                ntype = ['Lobatto' for _ in p]
                p_shape = (self._space.n, )

            else:
                raise NotImplementedError(f"cannot parse degree={degree}.")

        elif self._indicator == 'bundle':
            if isinstance(degree, (int, float)):
                assert degree % 1 == 0 and degree > 0, f"degree wrong."
                p = tuple([[degree for _ in range(self._space.n)] for _ in range(self._space.n)])
                nodes = [[] for _ in range(self._space.n)]
                ntype = [[] for _ in range(self._space.n)]
                for i in range(self._space.n):
                    for j in range(self._space.n):
                        nodes[i].append(
                            Quadrature(p[i][j], category='Lobatto').quad[0]
                        )
                        ntype[i].append('Lobatto')
                p_shape = (self._space.n, self._space.n)
            elif isinstance(degree, (list, tuple)) and all([isinstance(_, int) for _ in degree]):
                assert len(degree) == self._space.n, f"degree dimension wrong."
                p = tuple([degree for _ in range(self._space.n)])
                nodes = [[] for _ in range(self._space.n)]
                ntype = [[] for _ in range(self._space.n)]
                for i in range(self._space.n):
                    for j in range(self._space.n):
                        nodes[i].append(
                            Quadrature(p[i][j], category='Lobatto').quad[0]
                        )
                        ntype[i].append('Lobatto')
                p_shape = (self._space.n, self._space.n)

            else:
                raise NotImplementedError(f"cannot parse degree.p={degree} for `bundle` spaces.")

        else:
            raise NotImplementedError(self._indicator)

        self._p = p
        self._p_shape = p_shape
        self._nodes = nodes
        self._ntype = ntype

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
            if self._indicator in ('Lambda', 'bundle-diagonal'):
                self._edges = tuple([nodes[1:]-nodes[:-1] for nodes in self._nodes])
            elif self._indicator == 'bundle':
                s0, s1 = self._p_shape
                self._edges = [[] for _ in range(s0)]
                for i in range(s0):
                    for j in range(s1):
                        self._edges[i].append(
                            self._nodes[i][j][1:]-self._nodes[i][j][:-1]
                        )
            else:
                raise NotImplementedError(self._indicator)
        return self._edges

    @property
    def bfs(self):
        """1d basis functions."""
        if self._bfs is None:
            if self._indicator in ('Lambda', 'bundle-diagonal'):
                self._bfs = tuple([_OneDimPolynomial(nodes) for nodes in self.nodes])
            elif self._indicator == 'bundle':
                s0, s1 = self._p_shape
                self._bfs = [[] for _ in range(s0)]
                for i in range(s0):
                    for j in range(s1):
                        self._bfs[i].append(
                            _OneDimPolynomial(self._nodes[i][j])
                        )
            else:
                raise NotImplementedError(self._indicator)
        return self._bfs

    @property
    def num_local_dof_components(self):
        """"""
        return self._space.num_local_dof_components(self._degree)

    @property
    def num_local_dofs(self):
        """"""
        return self._space.num_local_dofs(self._degree)

    @property
    def gathering_matrix(self):
        """"""
        return self._space.gathering_matrix(self._degree)

    @property
    def incidence_matrix(self):
        return self._space.incidence_matrix(self._degree)

    @property
    def local_numbering(self):
        return self._space.local_numbering(self._degree)
