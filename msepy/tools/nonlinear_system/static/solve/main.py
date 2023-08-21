# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from msepy.tools.nonlinear_system.static.solve.Newton_Raphson import MsePyNonlinearSystemNewtonRaphsonSolve


class MsePyStaticNonlinearSystemSolve(Frozen):
    """"""
    def __init__(self, nls):
        """"""
        self._nls = nls
        self._bc = nls._bc
        self._scheme = 'Newton-Raphson'
        self._Newton_Raphson = MsePyNonlinearSystemNewtonRaphsonSolve(nls)
        self._message = ''
        self._info = None
        self._freeze()

    def _apply_bc(self):
        """"""
        if self._bc is None or len(self._bc) == 0:
            pass
        else:
            # by customizing the local static nonlinear system, we apply the bc.

            for boundary_section in self._bc:
                bcs = self._bc[boundary_section]
                for j, bc in enumerate(bcs):
                    number_application = bc._num_application

                    if number_application == 0:  # this particular not take effect yet

                        particular_bc = self._bc[boundary_section][j]

                        # applying it here
                        particular_bc.apply(self._nls)

                        particular_bc._num_application += 1

                    else:
                        assert number_application == 1

            # clean all number_application, to make sure in future static system, they are applied!
            # This is because this bc is from the dynamic linear system, and it is never changed.
            for boundary_section in self._bc:
                bcs = self._bc[boundary_section]
                for j, bc in enumerate(bcs):
                    bc._num_application = 0

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
        self._apply_bc()
        if self._scheme == 'Newton-Raphson':
            # the local cochain results is a list of 2d local cochains for self._nls.unknowns (already updated).
            local_cochain_results, message, info = self._Newton_Raphson(*args, **kwargs)
        else:
            raise NotImplementedError(f"I have no scheme called {self._scheme}.")

        self._message = message
        self._info = info

        return local_cochain_results
