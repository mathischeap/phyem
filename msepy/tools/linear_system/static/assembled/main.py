# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.tools.matrix.static.assembled import MsePyStaticAssembledMatrix
from msepy.tools.vector.static.assembled import MsePyStaticAssembledVector
from msepy.tools.linear_system.static.assembled.solve import MsePyStaticLinearSystemAssembledSolve


class MsePyStaticLinearSystemAssembled(Frozen):
    """Assembled system."""

    def __init__(self, A, b):
        """"""
        assert A.__class__ is MsePyStaticAssembledMatrix, f"A needs to be a {MsePyStaticAssembledMatrix}"
        assert b.__class__ is MsePyStaticAssembledVector, f"b needs to be a {MsePyStaticAssembledVector}"
        self._A = A
        self._b = b
        self._solve = MsePyStaticLinearSystemAssembledSolve(A, b)
        self._freeze()

    @property
    def A(self):
        """``A`` of ``Ax = b``."""
        return self._A

    @property
    def b(self):
        """``b`` of ``Ax = b``."""
        return self._b

    @property
    def solve(self):
        """Solve the system."""
        return self._solve

    @property
    def condition_number(self):
        """The condition number of A"""
        return self._A.condition_number

    @property
    def shape(self):
        return self.A.shape

    @property
    def rank(self):
        """The rank of A"""
        return self._A.rank

    @property
    def num_singularities(self):
        """Number of singularities in A"""
        return self._A.num_singularities

    def spy(self, *args, **kwargs):
        """spy plot of A"""
        self._A.spy(*args, **kwargs)
