# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class IrregularStaticLinearSystemCustomize(Frozen):
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

    def set_local_dof_ij_of_unknown_k_to_value(self, i, j, k, value):
        """Do what this method name means.

        Parameters
        ----------
        i :
            (i, j) refers `j`th local dof in #`i` cell of `k`th unknown.
        j :
            (i, j) refers `j`th local dof in #`i` cell of `k`th unknown.
        k :
            (i, j) refers `j`th local dof in #`i` cell of `k`th unknown.
        value

        Returns
        -------

        """
        row_gms = self._ls._row_gms
        assert j % 1 == 0 and j >= 0, f"local dof #{j} is out of range."
        before_gms = row_gms[:k]
        local_index_add = 0
        for _bgm in before_gms:
            local_index_add += len(_bgm[i])
        local_index = j + local_index_add
        global_row_gm = self._ls.global_gathering_matrices[0]
        global_numbering = global_row_gm[i][local_index]
        self.set_dof(int(global_numbering), value)
