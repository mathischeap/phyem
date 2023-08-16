# -*- coding: utf-8 -*-
"""
By Yi Zhang
Created at 6:33 PM on 8/13/2023
"""

from tools.frozen import Frozen


class MsePyNonlinearSystemCustomize(Frozen):
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

    def set_no_evaluation(self, i):
        """Let the nonlinear system do not affect the value of #r dof.

        So dof_i will be equal to dof_0 (the initial value (or initial guess)).
        """
        self._customizations.append(
            ('set_no_evaluation', i)
        )
