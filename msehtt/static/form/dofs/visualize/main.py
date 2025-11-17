# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.form.dofs.visualize.matplot import MseHtt_StaticForm_Dofs_Visualize_Matplot


class MseHtt_StaticForm_Dofs_Visualize(Frozen):
    r""""""
    def __init__(self, f):
        self._f = f
        self._matplot = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        r""""""
        return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        r""""""
        if self._matplot is None:
            self._matplot = MseHtt_StaticForm_Dofs_Visualize_Matplot(self._f)
        return self._matplot
