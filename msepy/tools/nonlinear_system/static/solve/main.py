# -*- coding: utf-8 -*-
"""
By Yi Zhang
Created at 6:34 PM on 8/13/2023
"""

from tools.frozen import Frozen
from msepy.tools.nonlinear_system.static.solve.Newton_Raphson import MsePyNonlinearSystemNewtonRaphsonSolve


class MsePyNonlinearSystemSolve(Frozen):
    """"""
    def __init__(self, nls):
        """"""
        self._nls = nls
        self._scheme = 'Newton-Raphson'
        self._Newton_Raphson = MsePyNonlinearSystemNewtonRaphsonSolve(nls)
        self._message = ''
        self._info = None
        self._freeze()

    @property
    def scheme(self):
        return self._scheme

    @property
    def message(self):
        """return the message of the last solver."""
        return self._message

    @property
    def info(self):
        """store the info of the last solver."""
        return self._info

    def __call__(self, *args, **kwargs):
        """Note that the results have updated the unknown cochains. Therefore, we do not have an option
         saying we update `x` or not.
         """
        if self._scheme == 'Newton-Raphson':
            results, message, info = self._Newton_Raphson(*args, **kwargs)
        else:
            raise NotImplementedError()

        self._message = message
        self._info = info

        return results
