# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.tools.vector.static.local import concatenate


class MseHttStaticLinearSystemCustomize(Frozen):
    """"""

    def __init__(self, sls):
        """"""
        self._sls = sls
        self._freeze()

    def set_dof(self, global_dof, value):
        """"""
        A = self._sls.A._mA
        b = self._sls.b._vb
        A.customize.identify_row(global_dof)
        b.customize.set_value(global_dof, value)

    def set_local_dof(self, ith_unknown, element_index, local_dof_index, value):
        r""""""
        A = self._sls.A._mA
        b = self._sls.b._vb
        global_numbering0 = A._gm_row.find_global_numbering_of_ith_composition_local_dof(
            ith_unknown, element_index, local_dof_index
        )
        global_numbering1 = b._gm.find_global_numbering_of_ith_composition_local_dof(
            ith_unknown, element_index, local_dof_index
        )
        assert global_numbering0 == global_numbering1
        A.customize.identify_row(global_numbering0)
        b.customize.set_value(global_numbering0, value)

    def set_element_only_local_dof(self, element_index, local_dof, value):
        """Only change the local system of element #`element_index`.

        This is useful, for example, when the problem is going to be solved element-wise.

        """
        A = self._sls.A._mA
        b = self._sls.b._vb
        A.customize.identify_local_dof(element_index, local_dof)
        b.customize.set_local_value(element_index, local_dof, value)

    def apply_essential_bc(self, ith_unknown, place, condition, time):
        """Apply essential bc to ith unknown on ``place`` with exact solution to be ``condition`` @ ``time``.

        Parameters
        ----------
        ith_unknown
        place
        condition
        time

        Returns
        -------

        """
        x = self._sls.x._x[ith_unknown]
        f = x._f
        gm = f.cochain.gathering_matrix
        global_dofs = place.find_dofs(f, local=False)

        local_cochain = f.reduce(condition @ time)
        global_cochain = gm.assemble(local_cochain, mode='replace')
        global_cochain = global_cochain[global_dofs]

        A_blocks = self._sls.A._A
        A_row = A_blocks[ith_unknown]
        b = self._sls.b._b[ith_unknown]

        for j, A in enumerate(A_row):
            if ith_unknown != j:
                A.customize.zero_rows(global_dofs)
            else:
                A.customize.identify_rows(global_dofs)

        b.customize.set_values(global_dofs, global_cochain)

    def left_matmul_A_block(self, i, j, M):
        """"""
        A = self._sls.A._A
        A[i][j] = M @ A[i][j]

    def left_matmul_b_block(self, i, M):
        """"""
        b = self._sls.b._b
        b[i] = M @ b[i]
        self._sls.b._vb = concatenate(b, self._sls.A._mA._gm_row)
