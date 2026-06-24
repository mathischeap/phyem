
from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.nonlinear_system.static_composite.solve.Newton_Raphson import (
    MseHtt_Composite_NonlinearSystem_NewtonRaphson_Solve)


class MseHtt_Static_Composite_Nonlinear_System_Solve(Frozen):
    """"""

    def __init__(self, c_nls):
        """"""
        self._c_nls = c_nls
        self._scheme = 'Newton-Raphson'
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

    def __call__(self, *args, scheme=None, **kwargs):
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
        if scheme is None:
            pass
        else:
            if scheme in ('Newton-Raphson', 'Newton'):
                self._scheme = 'Newton-Raphson'
            elif scheme == "Picard":
                self._scheme = 'Picard'
            else:
                raise NotImplementedError(f"I cannot do scheme={scheme}.")

        # ===========================================================================================

        static_nonlinear_systems = self._c_nls._static_nonlinear_systems

        for sns in static_nonlinear_systems:
            sns_solve = sns.solve

            if scheme is None:
                pass
            else:
                if scheme in ('Newton-Raphson', 'Newton'):
                    sns_solve._scheme = 'Newton-Raphson'
                elif scheme == "Picard":
                    sns_solve._scheme = 'Picard'
                else:
                    raise NotImplementedError(f"I cannot do scheme={scheme}.")

            # ------ take care of customizations on hold -------------------------------------------------
            customizations_on_hold = sns.customize.customizations_on_hold

            for cus in customizations_on_hold:
                indicator, cus_kwargs = cus
                if indicator == 'nonlinear_part_essential_bc':

                    ith_unknown = cus_kwargs['ith_unknown']
                    global_dofs = cus_kwargs['global_dofs']
                    global_cochain = cus_kwargs['global_cochain']

                    if self.scheme == 'Newton-Raphson':
                        sns.customize.fixed_global_dofs_for_unknown(ith_unknown, global_dofs)
                        sns.customize.set_x0_for_unknown(ith_unknown, global_dofs, global_cochain)
                    elif self.scheme == 'Picard':
                        sns.customize.set_global_dofs_for_unknown(
                            ith_unknown, global_dofs, global_cochain
                        )
                    else:
                        raise NotImplementedError()

                else:
                    raise NotImplementedError(f'customizations_on_hold indicator = {indicator}')

            # ------ below we check no nonlinear customization has taken effect yet ---------------------------
            nonlinear_customizations = sns.customize._nonlinear_customizations
            for i, setting in enumerate(nonlinear_customizations):
                customization_indicator = setting['customization_indicator']
                assert setting['take-effect'] == 0, \
                    f"{i}th nonlinear-customization {customization_indicator} must does not take any effect yet."

        # --------------------- solve it ----------------------------------------------------------------------
        if self.scheme == 'Newton-Raphson':
            # the local cochain results is a list of 2d local cochains for self._nls.unknowns (already updated).
            x, message, info = MseHtt_Composite_NonlinearSystem_NewtonRaphson_Solve(self._c_nls)(*args, **kwargs)
        else:
            raise NotImplementedError(f"I have no scheme called {self.scheme}.")

        # ------ below we check all the nonlinear customizations have taken effect -----------------------
        for sns in static_nonlinear_systems:
            nonlinear_customizations = sns.customize._nonlinear_customizations
            for i, setting in enumerate(nonlinear_customizations):
                customization_indicator = setting['customization_indicator']
                assert setting['take-effect'] == 1, \
                    f"{i}th nonlinear-customization <{customization_indicator}> does not (fully) take effect."
        # =================================================================================================

        self._message = message
        self._info = info

        return x, message, info
