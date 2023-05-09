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
            p = tuple([degree for _ in range(self._space.n)])
            nodes = tuple(
                [Quadrature(degree, category='Lobatto').quad[0] for _ in range(self._space.n)]
            )
            ntype = ['Lobatto' for _ in p]

        elif isinstance(degree, (list, tuple)) and all([isinstance(_, int) for _ in degree]):
            # for example, degree = (3, 2, ...)
            assert len(degree) == self._space.n, f"degree dimension wrong."
            p = tuple(self._degree)
            nodes = tuple(
                [Quadrature(_, category='Lobatto').quad[0] for _ in p]
            )
            ntype = ['Lobatto' for _ in p]

        else:
            raise NotImplementedError(f"cannot parse degree={degree}.")

        self._p = p
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
            self._edges = tuple([nodes[1:]-nodes[:-1] for nodes in self._nodes])
        return self._edges

    @property
    def bfs(self):
        """1d basis functions."""
        if self._bfs is None:
            self._bfs = tuple([_OneDimPolynomial(nodes) for nodes in self.nodes])
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
