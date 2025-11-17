# -*- coding: utf-8 -*-
r"""
"""
from scipy.sparse import linalg as spspalinalg
from time import time

from phyem.tools.frozen import Frozen
from phyem.tools.miscellaneous.timer import MyTimer


class _PackageScipy(Frozen):
    """"""
    def __init__(self):
        self._freeze()

    @staticmethod
    def spsolve(A, b, **kwargs):
        """direct solver."""
        if len(kwargs) > 0:
            print(f'warning: kwargs={kwargs} have no affects on scipy spsolve.')
        else:
            pass

        t_start = time()
        # ------------------------------------------------------------------
        x = spspalinalg.spsolve(A._M, b._v)
        # ==================================================================
        t_cost = time() - t_start
        t_cost = MyTimer.seconds2dhms(t_cost)
        message = f"Linear system of shape: {A.shape}" + \
            f" <direct solver costs: {t_cost}> "
        info = {
            'total cost': t_cost,
        }
        return x, message, info

    @staticmethod
    def lsqr(A, b, x0, **kwargs):
        """Compute least-squares solution to equation Ax = b.

        Compute a vector x such that the 2-norm |b - Ax| is minimized.
        """
        t_start = time()
        # -----------------------------------------------------------------
        results = spspalinalg.lsqr(A._M, b._v, x0=x0, **kwargs)
        # =================================================================
        x = results[0]
        t_cost = time() - t_start
        t_cost = MyTimer.seconds2dhms(t_cost)
        message = f"Linear system of shape: {A.shape}" + \
            f" <least square solver costs: {t_cost}> "
        info = {
            'total cost': t_cost,
            'solver info': results[1:-1]
        }
        return x, message, info

    def _parse_preconditioner(self, A, preconditioner_parameters):
        """make a precondition according to the parameters.

        Support:
            1) name: spilu
                parameters:
                    drop_tol: float, optional,
                        Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition. (default: 1e-4)
                    fill_factor： float, optional
                        Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)

                So, do for example,
                preconditioner_parameters = {
                    'name': 'spilu',
                    'drop_tol': 1e-4,
                    'fill_factor': 10,
                }

        """
        if preconditioner_parameters is None:
            return None
        else:
            if preconditioner_parameters is True:  # use the default preconditioner: spilu
                preconditioner_parameters = {
                    'name': 'spilu',
                    'drop_tol': 1e-4,
                    'fill_factor': 10,
                }
            else:
                pass

            assert 'name' in preconditioner_parameters, \
                f"'name' key must be in preconditioner_parameters to indicate the preconditioner type."
            preconditioner_name = preconditioner_parameters['name']
            parameters = dict()
            for key in preconditioner_parameters:
                if key != 'name':
                    parameters[key] = preconditioner_parameters[key]
                else:
                    pass
            # --- below we make the preconditioner --------------------------------------------
            if preconditioner_name == 'spilu':
                ILU_fact = spspalinalg.spilu(A, **parameters)
                # noinspection PyArgumentList
                M = spspalinalg.LinearOperator(
                    shape=A.shape,
                    matvec=lambda b: ILU_fact.solve(b)
                )
                return M

            else:
                raise NotImplementedError(f"cannot make preconditioner {preconditioner_name}.")
            # =================================================================================

    def gmres(self, A, b, x0, preconditioner=None, **kwargs):
        """

        Parameters
        ----------
        A
        b
        x0
        preconditioner :
            None, True, or use a particular preconditioner by, for example,
                preconditioner = {
                    'name': p_name,
                    'para1_name': p1,
                    ...,
                }
            When it is None, no preconditioner; when it is True, use the default preconditioner.
        kwargs

        Returns
        -------

        """
        t_start = time()

        # ------------------------------------------------------------
        A = A._M
        b = b._v
        # ============================================================

        preconditioner = self._parse_preconditioner(A, preconditioner)

        x, info = spspalinalg.gmres(
            A, b, x0=x0, M=preconditioner, **kwargs
        )

        t_cost = time() - t_start
        t_cost = MyTimer.seconds2dhms(t_cost)
        info_kwargs_exclusive = ['x0', 'M', 'callback']
        info_kwargs = {
            'preconditioner': preconditioner,
        }
        for key in kwargs:
            if key not in info_kwargs_exclusive:
                info_kwargs[key] = kwargs[key]
        message = f"Linear system of shape: {A.shape} " + \
            f"<gmres costs: {t_cost}> <info: {info}> <inputs: {info_kwargs}>"
        info = {
            'total cost': t_cost,
            'convergence info': info,
        }
        return x, message, info

    def lgmres(self, A, b, x0, preconditioner=False, **kwargs):
        """"""
        t_start = time()

        # ------------------------------------------------------------
        A = A._M
        b = b._v
        # ============================================================

        preconditioner = self._parse_preconditioner(A, preconditioner)

        x, info = spspalinalg.lgmres(
            A, b, x0=x0, M=preconditioner, **kwargs
        )
        t_cost = time() - t_start
        t_cost = MyTimer.seconds2dhms(t_cost)
        info_kwargs_exclusive = ['x0', 'M', 'callback']
        info_kwargs = {
            'preconditioner': preconditioner,
        }
        for key in kwargs:
            if key not in info_kwargs_exclusive:
                info_kwargs[key] = kwargs[key]
        message = f"Linear system of shape: {A.shape} " + \
            f"<lgmres costs: {t_cost}> <info: {info}> <inputs: {info_kwargs}>"
        info = {
            'total cost': t_cost,
            'convergence info': info,
        }
        return x, message, info
