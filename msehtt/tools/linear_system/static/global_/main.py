# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.matrix.static.global_ import MseHttGlobalMatrix
from phyem.msehtt.tools.vector.static.global_distributed import MseHttGlobalVectorDistributed
from phyem.msehtt.tools.linear_system.static.global_.solve import MseHttLinearSystemSolve


class MseHttLinearSystem(Frozen):
    r""""""

    def __init__(self, A, b):
        r""""""
        assert A.__class__ is MseHttGlobalMatrix, f"A must be {MseHttGlobalMatrix}. Now it is {A.__class__}"
        assert b.__class__ is MseHttGlobalVectorDistributed, f"b must be {MseHttGlobalVectorDistributed}."
        A_shape = A.shape
        b_shape = b.shape
        assert A_shape[0] == b_shape[0], f"A, b shape dis-match."
        self._A = A
        self._b = b
        A_gm_row, A_gm_col = A.gm_row, A.gm_col
        b_gm = b.gm
        if A_gm_row is None and b_gm is None:
            gm_row = None
        elif A_gm_row is None:
            gm_row = b_gm
        elif b_gm is None:
            gm_row = A_gm_row
        else:
            assert A_gm_row == b_gm, f"A-row-gm and b-gm do not match."
            gm_row = A_gm_row
        self._gm_row = gm_row
        self._gm_col = A_gm_col
        self._solve = None
        self._freeze()

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<msehtt static global linear system of shape {self.shape} at" + super_repr

    @property
    def A(self):
        r"""`A` of `Ax=b`."""
        return self._A

    @property
    def b(self):
        r"""`b` of `Ax=b`."""
        return self._b

    @property
    def shape(self):
        r"""The shape of Ax=b, i.e. the shape of A."""
        return self.A.shape

    @property
    def gm_row(self):
        r"""The row gathering matrix."""
        return self._gm_row

    @property
    def gm_col(self):
        r"""The column gathering matrix."""
        return self._gm_col

    @property
    def solve(self):
        r"""the solving part."""
        if self._solve is None:
            self._solve = MseHttLinearSystemSolve(self)
        return self._solve

    @property
    def condition_number(self):
        r""""""
        return self.A.condition_number

    @property
    def rank(self):
        r""""""
        return self.A.rank

    @property
    def rank_nnz(self):
        r""""""
        return self.A.rank_nnz

    @property
    def num_singularities(self):
        r""""""
        return self.A.num_singularities

    def spy(self, **kwargs):
        r""""""
        return self.A.spy(**kwargs)
