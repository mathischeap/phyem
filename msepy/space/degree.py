# -*- coding: utf-8 -*-
r"""
How to parse a degree for Mse Py space.
"""
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from msepy.tools.polynomials import _OneDimPolynomial


class PySpaceDegree(Frozen):
    """"""

    def __init__(self, space, degree):
        """"""
        self._space = space
        self._n = space.n
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
                p = tuple([degree for _ in range(self._n)])
                nodes = tuple(
                    [Quadrature(degree, category='Lobatto').quad[0] for _ in range(self._n)]
                )
                ntype = ['Lobatto' for _ in p]
                p_shape = (self._n, )

            elif isinstance(degree, (list, tuple)) and all([isinstance(_, int) for _ in degree]):
                # for example, degree = (3, 2, ...)
                assert len(degree) == self._n, f"degree dimension wrong."
                p = tuple(self._degree)
                nodes = tuple(
                    [Quadrature(_, category='Lobatto').quad[0] for _ in p]
                )
                ntype = ['Lobatto' for _ in p]
                p_shape = (self._n, )

            else:
                raise NotImplementedError(f"cannot parse degree={degree}.")

        elif self._indicator == 'bundle':
            if isinstance(degree, (int, float)):
                assert degree % 1 == 0 and degree > 0, f"degree wrong."
                p = tuple([[degree for _ in range(self._n)] for _ in range(self._n)])
                nodes = [[] for _ in range(self._n)]
                ntype = [[] for _ in range(self._n)]
                for i in range(self._n):
                    for j in range(self._n):
                        nodes[i].append(
                            Quadrature(p[i][j], category='Lobatto').quad[0]
                        )
                        ntype[i].append('Lobatto')
                p_shape = (self._n, self._n)
            elif isinstance(degree, (list, tuple)) and all([isinstance(_, (list, tuple)) for _ in degree]):
                assert len(degree) == self._n, f"degree dimension wrong."
                for i, Di in enumerate(degree):
                    assert len(Di) == self._n, f"degree dimension wrong."
                    for j, pij in enumerate(Di):
                        assert isinstance(pij, (int, float)), f"degree[{i}][{j}] = {pij} wrong, it must be a number."
                        assert pij % 1 == 0 and pij > 0, \
                            f"degree[{i}][{j}] = {pij} wrong, it must be a positive integer."
                p = degree
                nodes = [[] for _ in range(self._n)]
                ntype = [[] for _ in range(self._n)]
                for i in range(self._n):
                    for j in range(self._n):
                        nodes[i].append(
                            Quadrature(p[i][j], category='Lobatto').quad[0]
                        )
                        ntype[i].append('Lobatto')
                p_shape = (self._n, self._n)

            else:
                raise NotImplementedError(f"cannot parse degree.p={degree} for `bundle` spaces.")

        else:
            raise NotImplementedError(self._indicator)

        self._p = p
        self._p_shape = p_shape
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
        """For msehy-py spaces, access gathering matrix here only for more recent generation mesh!"""
        from msepy.mesh.main import MsePyMesh
        mesh_class = self._space.mesh.__class__
        if mesh_class is MsePyMesh:
            return self._space.gathering_matrix(self._degree)
        else:
            raise Exception()

    @property
    def incidence_matrix(self):
        return self._space.incidence_matrix(self._degree)

    @property
    def local_numbering(self):
        return self._space.local_numbering(self._degree)

    def inner_product(self, other, special_key=None):
        """"""
        if other.__class__ is self.__class__:
            return self._space.inner_product(self._degree, other, special_key=special_key)
        else:
            raise NotImplementedError()
