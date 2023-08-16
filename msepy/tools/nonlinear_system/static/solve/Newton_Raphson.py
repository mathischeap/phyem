# -*- coding: utf-8 -*-
"""
By Yi Zhang
Created at 6:34 PM on 8/13/2023
"""
import numpy as np
from time import time
from tools.frozen import Frozen
from msepy.form.main import MsePyRootForm
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.tools.linear_system.static.local import MsePyStaticLocalLinearSystem


class MsePyNonlinearSystemNewtonRaphsonSolve(Frozen):
    """"""
    def __init__(self, nls):
        """"""
        self._nls = nls
        self._x0 = None
        self._freeze()

    @property
    def x0(self):
        """the initial guess for iterative solvers."""
        if self._x0 is None:
            raise Exception(f"x0 is None, first set it.")
        return self._x0

    @x0.setter
    def x0(self, _x0):
        """"""
        if all([_.__class__ is MsePyRootForm for _ in _x0]):   # providing all MsePyRootForm
            # use the newest cochains.
            cochain = list()

            for f in _x0:
                newest_time = f.cochain.newest
                gm = f.cochain.gathering_matrix
                if newest_time is None:  # no newest cochain at all.
                    # then use 0-cochain
                    local_cochain = np.zeros_like(gm._gm)

                else:
                    local_cochain = f.cochain[newest_time].local

                cochain.append(local_cochain)

            self._x0 = cochain   # a list of 2d-array; the local cochains.

        else:
            raise NotImplementedError()

    def __call__(
            self,
            x0, atol=1e-4, maxiter=10,        # args for the outer Newton method.
            # for the inner solver
            inner_solver_package='scipy',
            inner_solver_scheme='spsolve',
            inner_solver_kwargs=None          # args for the inner linear solver.
    ):
        """"""

        t_start = time()

        if inner_solver_kwargs is None:
            inner_solver_kwargs = dict()
        else:
            pass

        # parse x0 from the given x0, usually a list of forms.
        self.x0 = x0

        # initialize the variables ---------------------------------------------------------------- 1

        xi = self.x0   # a list of 2d arrays;
        ITER = 0
        message = ''
        BETA = list()

        t_iteration_start = time()

        while 1:
            ITER += 1

            itmV = xi  # send xi to intermediate values

            # --- evaluate blocks of the left-hand-side matrix with xi ---------------------------- 2

            S0, S1 = self._nls.shape
            LHS = [[None for _ in range(S1)] for _ in range(S0)]

            for i in range(S0):

                if i in self._nls._nonlinear_terms:
                    NTi = self._nls._nonlinear_terms[i]
                    NSi = self._nls._nonlinear_signs[i]
                else:
                    NTi = None
                    NSi = None

                test_form = self._nls.test_forms[i]

                for j in range(S1):

                    # __________ we first find the linear term from A ______________________ 3

                    linear_term = self._nls._A[i][j]
                    if linear_term is not None:
                        assert linear_term.__class__ is MsePyStaticLocalMatrix, \
                            f"A linear term [{i}][{j}] must be a {MsePyStaticLocalMatrix},"

                        assert LHS[i][j] is None, f"LHS[i][j] must be untouched yet!"
                        LHS[i][j] = linear_term
                    else:
                        pass

                    # __ we then look at the nonlinear terms to find the contributions ____ 4

                    if NTi is None:
                        pass

                    else:
                        unknown_form = self._nls.unknowns[j]

                        for term, sign in zip(NTi, NSi):
                            assert test_form in term.correspondence
                            if unknown_form in term.correspondence:

                                known_pairs = list()

                                for cp in term.correspondence:
                                    if cp not in (unknown_form, test_form):

                                        itmV_cp = None
                                        for k, uk in enumerate(self._nls.unknowns):
                                            if cp == uk:
                                                itmV_cp = itmV[k]
                                                break
                                            else:
                                                pass

                                        known_pairs.append(
                                            [cp, itmV_cp]
                                        )
                                    else:
                                        pass

                                contribution_2_Aij = term._derivative_contribution(
                                    test_form,
                                    unknown_form,
                                    *known_pairs
                                )

                                assert contribution_2_Aij.__class__ is MsePyStaticLocalMatrix, \
                                    (f"contribution of nonlinear term to linear system must be a "
                                     f"{MsePyStaticLocalMatrix},")

                                if sign == '+':
                                    pass
                                else:
                                    contribution_2_Aij = - contribution_2_Aij

                                # we have found a nonlinear contribution for lhs[i][j], add it to it.
                                LHS[i][j] += contribution_2_Aij

                            else:
                                pass

            f = self._nls.evaluate_f(itmV, neg=True)  # notice the neg here!

            x = list()
            for uk in self._nls.unknowns:
                x.append(uk._f.cochain.static_vec(uk._t))

            ls = MsePyStaticLocalLinearSystem(LHS, x, f)

            # ---------------- adopt customizations ------------------------------------------2
            for customization in self._nls.customize._customizations:
                method_name = customization[0]
                method_para = customization[1]

                A, f = ls.A._mA, ls.b._vb
                # ------------------------------------------------------------3
                if method_name == "set_no_evaluation":
                    # we make the #i dof unchanged .....
                    A.customize.identify_row(method_para)
                    f.customize.set_value(method_para, 0)

                else:
                    raise NotImplementedError(f"Cannot handle customization = {method_name}")

            als = ls.assemble()
            solve = als.solve
            solve.package = inner_solver_package
            solve.scheme = inner_solver_scheme
            solve.x0 = 0
            A_shape = solve._A.shape

            solve(update_x=True, **inner_solver_kwargs)  # results updated to the unknowns of the nonlinear system
            LSm = solve.message

            dx = list()
            for f in self._nls.unknowns:
                dx.append(f.cochain.local)

            beta = sum(
                [
                    np.sum(_**2) for _ in dx
                ]
            )
            BETA.append(beta)
            JUDGE, stop_iteration, convergence_info, JUDGE_explanation = \
                _nLS_stop_criterion(BETA, atol, ITER, maxiter)

            xi1 = list()
            for _xi, _dx in zip(xi, dx):
                xi1.append(
                    _xi + _dx
                )

            if stop_iteration:
                break
            else:
                xi = xi1

        # ------ Newton iteration completed xi1 is the solution...
        results = xi1
        for k, uk in enumerate(self._nls.unknowns):
            uk.cochain = results[k]  # the results will be sent to the unknowns.

        t_iteration_end = time()
        Ta = t_iteration_end-t_start
        TiT = t_iteration_end-t_iteration_start

        message += f"<nonlinear_solver>" \
                   f" = [RegularNewtonRaphson: {A_shape}]" \
                   f" of {self._nls._num_nonlinear_terms} nonlinear terms" \
                   f" : atol={atol}, maxiter={maxiter} + Linear solver: {inner_solver_package} {inner_solver_scheme}" \
                   f" args: {inner_solver_kwargs}" \
                   f" -> [ITER: {ITER}]" \
                   f" = [beta: %.4e]" \
                   f" = [{convergence_info}-{JUDGE_explanation}]" \
                   f" -> nLS solving costs %.2f, each ITER cost %.2f" % (BETA[-1], Ta, TiT/ITER) \
                   + '\n --- Last Linear Solver Message: \n' + LSm

        info = {
            'total cost': Ta,
            'convergence info': convergence_info,
            'iteration cost': TiT/ITER,
            'residuals': BETA,
        }

        return results, message, info


def _nLS_stop_criterion(BETA, atol, ITER, maxiter):
    """

    Parameters
    ----------
    BETA
    atol
    ITER
    maxiter : int, str
        If it is a str, then it is a forced maxiter, which is the only criterion of stopping
        the iterations.

    Returns
    -------

    """

    if isinstance(maxiter, str):

        MAXITER = int(maxiter)
        judge_2 = ITER >= MAXITER  # judge 2: reach max iteration number

        judge_1 = False
        judge_3 = False
        judge_4 = False

    else:

        beta = BETA[-1]
        judge_1 = beta < atol  # judge 1: reach absolute tolerance.
        judge_2 = ITER >= maxiter  # judge 2: reach max iteration number

        # judge 3: divergence
        if (not judge_1) and len(BETA) > 1 and BETA[-1] > BETA[-2]:  # error grows after one iteration
            if BETA[-2] > 1 and (BETA[-1] - BETA[-2]) > 0.5 * BETA[-2]:
                judge_3 = True
            elif BETA[-1] > 10e6:
                judge_3 = True
            elif (BETA[-1] - BETA[-2]) > 10:
                judge_3 = True
            elif BETA[-1] > 10 * BETA[-2]:
                judge_3 = True
            else:
                judge_3 = False
        else:
            judge_3 = False

        # judge_4: slow converging
        if (not judge_1) and (len(BETA) > 1) and (beta < BETA[-2]):
            beta_old = BETA[-2]
            progress = beta_old - beta
            if progress / beta_old < 0.0001:  # slow converging
                judge_4 = True
            else:
                judge_4 = False
        else:
            judge_4 = False

    # -------------------------------------------------------------------------------
    if judge_1 or judge_2 or judge_3 or judge_4:

        stop_iteration = True

        if judge_1:  # reach atol
            info = 0
            JUDGE = 1
            JUDGE_explanation = 'reach absolute tol'

        elif judge_2:  # reach maxiter
            info = ITER
            JUDGE = 2
            JUDGE_explanation = 'reach maxiter'

        elif judge_3:  # diverging
            info = -1
            JUDGE = 3
            JUDGE_explanation = 'diverging'

        elif judge_4:  # very slow converging;
            info = ITER
            JUDGE = 4
            JUDGE_explanation = 'very slow converging'
        else:
            raise Exception()

    else:  # do not stop iterations.
        stop_iteration = False
        info = None
        JUDGE = 0
        JUDGE_explanation = ''

    assert stop_iteration in (True, False), "stop_iteration has to be set."

    return JUDGE, stop_iteration, info, JUDGE_explanation
