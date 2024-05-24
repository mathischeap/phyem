# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehtt.tools.nonlinear_system.static.solve.Newton_Raphson import MseHttNonlinearSystemNewtonRaphsonSolve


class MseHttStaticNonlinearSystemSolve(Frozen):
    """"""

    def __init__(self, nls):
        """"""
        self._nls = nls
        self._scheme = 'Newton-Raphson'
        self._Newton_Raphson = MseHttNonlinearSystemNewtonRaphsonSolve(nls)
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
        """Note that the results have updated the unknown cochains.
        Therefore, we do not have an option saying we update `x` or not.

        Returns
        -------
        x : list
            A list of solutions of nls.unknowns. So, they have been distributed to
            local unknowns.
        message : str
            A string message.
        info : Dict[str]
            A dictionary of information.

        """
        # ------ below we check no nonlinear customization has taken effect yet ---------------------------
        nonlinear_customizations = self._nls.customize._nonlinear_customizations
        for i, setting in enumerate(nonlinear_customizations):
            customization_indicator = setting['customization_indicator']
            assert setting['take-effect'] == 0, \
                f"{i}th nonlinear-customization {customization_indicator} must does not take any effect yet."

        # --------------------- solve it ----------------------------------------------------------------------
        if self._scheme == 'Newton-Raphson':
            # the local cochain results is a list of 2d local cochains for self._nls.unknowns (already updated).
            x, message, info = self._Newton_Raphson(*args, **kwargs)
        else:
            raise NotImplementedError(f"I have no scheme called {self._scheme}.")

        # ------ below we check all the nonlinear customizations have taken effect -----------------------
        nonlinear_customizations = self._nls.customize._nonlinear_customizations
        for i, setting in enumerate(nonlinear_customizations):
            customization_indicator = setting['customization_indicator']
            assert setting['take-effect'] == 1, \
                f"{i}th nonlinear-customization {customization_indicator} does not fully take effect."
        # =================================================================================================

        self._message = message
        self._info = info

        return x, message, info
