# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen


class MsePyStaticNonlinearSystemCustomize(Frozen):
    """"""

    def __init__(self, nls):
        """"""
        self._nls = nls
        self._customizations = list()
        self._freeze()

    @property
    def customization(self):
        """customization."""
        return self._customizations

    def clear(self):
        """"""
        self._customizations = list()

    def set_no_evaluation(self, dof):
        """Let the nonlinear system do not affect the value of global #r dof.

        So cochain of `dof` will be equal to its x0 value (the initial value (or initial guess)).
        """
        self._customizations.append(
            ('set_no_evaluation', dof)
        )

    def set_x0_from_local_dofs(self, i, elements, local_dofs, local_values):
        """Set x0 for the #`i` unknown at its local positions (indicated by `elements` and `local_dofs`)
        to be `local values`
        """
        self._customizations.append(
            ('set_x0_from_local_dofs', [i, elements, local_dofs, local_values])
        )

    def set_no_evaluation_of_local_dofs(self, i, elements, local_dofs):
        """Let the local dofs (indicating by `elements` and `local_dofs`) of the #`i` unknown do not change.
        """
        unknowns = self._nls.unknowns
        local_dofs_base = 0
        if i > 0:
            for j in range(i):
                local_dofs_base += unknowns[j]._f.cochain.gathering_matrix.num_local_dofs

            overall_local_dofs = list()   # consider the local system as one matrix

            for ld in local_dofs:
                overall_local_dofs.append(
                    ld + local_dofs_base
                )

        else:
            overall_local_dofs = local_dofs

        self._customizations.append(
            ('set_no_evaluation_for_overall_local_dofs', [elements, overall_local_dofs])
        )
