# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import COMM


class MseHttStaticNonlinearSystemCustomize(Frozen):
    """"""

    def __init__(self, nls):
        """"""
        self._nls = nls
        self._nonlinear_customizations = list()
        self._linear = MseHttStaticNonlinearSystemCustomize_LinearPart(nls)
        self._freeze()

    @property
    def linear(self):
        return self._linear

    def ___check_nonlinear_customization___(self, customization_indicator, *args, **kwargs):
        """"""
        if customization_indicator == 'fixed_global_dofs_for_unknown':
            assert kwargs == {}, f"fixed_global_dofs_for_unknown customization accepts no kwargs."
            ith_unknown, global_dofs = args
            self._nonlinear_customizations.append(
                {
                    'customization_indicator': 'fixed_global_dofs_for_unknown',
                    'ith_unknown': ith_unknown,
                    'global_dofs': global_dofs,
                    'take-effect': 0,
                }
            )
        elif customization_indicator == 'set_x0_for_unknown':
            assert kwargs == {}, f"set_x0_for_unknown customization accepts no kwargs."
            ith_unknown, global_dofs, global_cochain = args
            self._nonlinear_customizations.append(
                {
                    'customization_indicator': 'set_x0_for_unknown',
                    'ith_unknown': ith_unknown,
                    'global_dofs': global_dofs,
                    'global_cochain': global_cochain,
                    'take-effect': 0,
                }
            )
        else:
            raise NotImplementedError(f"cannot accept customization_indicator={customization_indicator}.")

    def fixed_global_dofs_for_unknown(self, ith_unknown, global_dofs):
        """"""
        self.___check_nonlinear_customization___(
            'fixed_global_dofs_for_unknown', ith_unknown, global_dofs
        )

    def set_x0_for_unknown(self, ith_unknown, global_dofs, global_cochain):
        """"""
        self.___check_nonlinear_customization___(
            'set_x0_for_unknown', ith_unknown, global_dofs, global_cochain
        )


class MseHttStaticNonlinearSystemCustomize_LinearPart(Frozen):
    """"""

    def __init__(self, nls):
        """"""
        self._nls = nls
        self._freeze()

    def left_matmul_A_block(self, i, j, M):
        """If Aij = A[i][j], we will make A[i][j] become M @ Aij.

        Parameters
        ----------
        i
        j
        M

        Returns
        -------

        """
        A = self._nls._A
        A[i][j] = M @ A[i][j]

    def set_local_dof(self, ith_unknown, element_label, local_dof_index, value):
        """"""
        gm = self._nls._row_gms[ith_unknown]

        if element_label in gm:
            unknown_global_numbering = gm[element_label][local_dof_index]
        else:
            unknown_global_numbering = -1
        unknown_global_numbering = COMM.allgather(unknown_global_numbering)
        unknown_global_numbering = max(unknown_global_numbering)

        A = self._nls._A
        b = self._nls._b

        Ai_ = A[ith_unknown]
        bi = b[ith_unknown]

        for j, Aij in enumerate(Ai_):
            if ith_unknown != j:
                Aij.customize.zero_row(unknown_global_numbering)
            else:
                Aij.customize.identify_row(unknown_global_numbering)

        bi.customize.set_value(unknown_global_numbering, value)

    def apply_essential_bc_for_unknown(self, ith_unknown, global_dofs, global_cochain):
        """"""
        A = self._nls._A
        b = self._nls._b

        Ai_ = A[ith_unknown]
        bi = b[ith_unknown]

        for j, Aij in enumerate(Ai_):
            if ith_unknown != j:
                Aij.customize.zero_rows(global_dofs)
            else:
                Aij.customize.identify_rows(global_dofs)

        bi.customize.set_values(global_dofs, global_cochain)
