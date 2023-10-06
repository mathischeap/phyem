# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from generic.py.linear_system.globalize.solve import Solve
from generic.py.matrix.globalize.static import Globalize_Static_Matrix
from generic.py.vector.globalize.static import Globalize_Static_Vector


class Globalize_Static_Linear_System(Frozen):
    """"""

    def __init__(self, A, b):
        """"""
        assert A.__class__ is Globalize_Static_Matrix, f"A needs to be a {Globalize_Static_Matrix}"
        assert b.__class__ is Globalize_Static_Vector, f"b needs to be a {Globalize_Static_Vector}"
        self._A = A
        self._b = b
        self._solve = Solve(A, b)
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
