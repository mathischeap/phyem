# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class _MPI_PY_LS_Customize(Frozen):
    """"""

    def __init__(self, ls):
        """"""
        self._ls = ls
        self._freeze()

    def set_dof(self, i, value):
        """Set the solution of dof #i to be `value`.

        Always remember the way how we chain multiple gathering matrices. So usually only i = 0, or i = -1
        refer to what we intuitively think it does.
        """
        A = self._ls.A._mA
        b = self._ls.b._vb
        A.customize.identify_row(i)
        b.customize.set_value(i, value)
