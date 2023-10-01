# -*- coding: utf-8 -*-
r"""
"""
from msehy.tools.matrix.static.assembled import IrregularStaticAssembledMatrix
from msehy.tools.vector.static.assembled import IrregularStaticAssembledVector
from msehy.tools.linear_system.static.assembled.solve import IrregularStaticLinearSystemAssembledSolve

from msepy.tools.linear_system.static.assembled.main import MsePyStaticLinearSystemAssembled


class IrregularStaticLinearSystemAssembled(MsePyStaticLinearSystemAssembled):
    """Assembled system."""

    def _check_Ab_and_initialize_solve(self, A, b):
        assert isinstance(A, IrregularStaticAssembledMatrix) and isinstance(b, IrregularStaticAssembledVector), \
            f"A or b type wrong."
        self._solve = IrregularStaticLinearSystemAssembledSolve(self)
