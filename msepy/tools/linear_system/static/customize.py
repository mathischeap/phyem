# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen


class MsePyStaticLinearSystemCustomize(Frozen):
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

    def set_local_dof(self, unknown_no, element, local_numbering, value):
        """Set the dof at local position #local_numbering of #element of #unknown_no unknown to be value.

        Parameters
        ----------
        unknown_no : {0, 1, 2}
            0 is the first unknowns.
        element
        local_numbering
        value

        Returns
        -------

        """
        local_numbering_start = 0
        gms = self._ls.gathering_matrices[0]
        for i in range(unknown_no):
            gm = gms[i]
            local_numbering_start += gm.num_local_dofs

        total_local_numbering = local_numbering_start + local_numbering
        global_numbering = self._ls.global_gathering_matrices[0][element, total_local_numbering]
        global_numbering = int(global_numbering)
        self.set_dof(global_numbering, value)

    def set_local_dof_ij_of_unknown_k_to_value(self, i, j, k, value):
        """Do what this method name means.

        Parameters
        ----------
        i :
            (i, j) refers `j`th local dof in #`i` element of `k`th unknown.
        j :
            (i, j) refers `j`th local dof in #`i` element of `k`th unknown.
        k :
            (i, j) refers `j`th local dof in #`i` element of `k`th unknown.
        value

        Returns
        -------

        """
        row_gms = self._ls._row_gms
        k_gm = row_gms[k]
        num_elements, k_num_local_dofs = k_gm.shape

        if i < 0:
            _i = i + k_num_local_dofs
        else:
            _i = i

        assert _i % 1 == 0 and 0 <= _i < num_elements, f"element #{i} is out of range."

        if j < 0:
            _j = j + k_num_local_dofs
        else:
            _j = j

        assert _j % 1 == 0 and 0 <= _j < k_num_local_dofs, f"local dof #{j} is out of range."

        before_gms = row_gms[:k]
        local_index_add = 0
        for _bgm in before_gms:
            local_index_add += _bgm.shape[1]
        local_index = _j + local_index_add
        global_row_gm = self._ls.global_gathering_matrices[0]
        global_numbering = global_row_gm._gm[_i, local_index]
        self.set_dof(int(global_numbering), value)
