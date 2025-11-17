# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.form.dofs.visualize.main import MseHtt_StaticForm_Dofs_Visualize


class MseHtt_StaticForm_Dofs(Frozen):
    r""""""

    def __init__(self, f):
        """"""
        if f._is_base():
            self._f = f
        else:
            raise Exception(f"dofs property can only be defined for the base form")
        self._visualize = None
        self._freeze()

    @property
    def num_rank_dofs(self):
        r"""Return the number of dofs in local rank."""
        return self._f.cochain.gathering_matrix.num_rank_dofs

    @property
    def num_global_dofs(self):
        r"""Return total number of dofs across all ranks in all ranks."""
        return self._f.cochain.num_global_dofs

    @property
    def visualize(self):
        r"""Visualize the dofs."""
        if self._visualize is None:
            self._visualize = MseHtt_StaticForm_Dofs_Visualize(self._f)
        return self._visualize

    def ___global_dof_info___(self):
        r"""Return a dict that contain the global dof information for all rank dofs."""
        local_dof_info = self._f.space.local_dofs(self._f.degree)
        gm = self._f.cochain.gathering_matrix
        global_dof_info = {}

        for e in gm:
            gme = gm[e]
            local_dofs = local_dof_info[e]
            for i, dof in enumerate(gme):
                global_dof_info[dof] = local_dofs[i]

        return global_dof_info
