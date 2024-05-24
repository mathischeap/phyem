# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


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
