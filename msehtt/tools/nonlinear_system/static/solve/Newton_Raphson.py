# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from time import time
from tools.frozen import Frozen
from msehtt.static.form.main import MseHttForm
from msehtt.tools.vector.static.local import MseHttStaticLocalVector
from msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from msehtt.tools.linear_system.static.local.main import MseHttStaticLocalLinearSystem


class MseHttNonlinearSystemNewtonRaphsonSolve(Frozen):
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
        if all([_.__class__ is MseHttForm for _ in _x0]):   # providing all MsePyRootForm
            # use the newest cochains.
            cochain = list()

            for f in _x0:
                newest_time = f.cochain.newest
                if newest_time is None:  # no newest cochain at all.
                    # then use 0-cochain
                    local_cochain = MseHttStaticLocalVector(0, f.cochain.gathering_matrix)
                else:
                    local_cochain = f.cochain[newest_time]
                cochain.append(local_cochain)
            self._x0 = cochain   # a list of 2d-array; the local cochains.

        else:
            raise NotImplementedError()

    def __call__(
            self,
            x0, atol=1e-4, maxiter=5,        # args for the outer Newton method.
            preconditioner=False,
            threshold=None,
            # for the inner solver below ------------------------------------------------
            inner_solver_scheme='spsolve',
            inner_solver_kwargs=None          # args for the inner linear solver.
    ):
        r"""

        Parameters
        ----------
        x0 :
            The initial guess. Usually be a list of forms. We will parse x0 from the newest cochain of
            these forms.
        atol :
            Outer Newton iterations stop when norm(dx) < atol.
        maxiter :
            Outer Newton iterations stop when ITER > maxiter.
        preconditioner :
        threshold :
        inner_solver_scheme :
        inner_solver_kwargs :

        Returns
        -------

        """

        t_start = time()

        if inner_solver_kwargs is None:
            inner_solver_kwargs = dict()
        else:
            pass

        # parse x0 from the given x0, usually a list of forms.
        self.x0 = x0

        # initialize the variables ---------------------------------------------------------------- 1

        x0 = self.x0
        ITER = 0
        message = ''
        BETA = list()

        # --------------- iterations start --------------------------------------------------
        t_iteration_start = time()

        # ------ some customizations need to be applied here --------------------------------
        for nonlinear_customization in self._nls.customize._nonlinear_customizations:
            indicator = nonlinear_customization['customization_indicator']

            # ----------------------------------------------------------------------
            if indicator == 'set_x0_for_unknown':
                assert nonlinear_customization['take-effect'] == 0  # it takes no effect yet.
                nonlinear_customization['take-effect'] = 1   # it takes fully effect here!
                ith_unknown = nonlinear_customization['ith_unknown']
                global_dofs = nonlinear_customization['global_dofs']
                global_cochain = nonlinear_customization['global_cochain']
                xi = x0[ith_unknown]
                new_xi = {}
                gm = self._nls._x[ith_unknown]._f.cochain.gathering_matrix
                local_positions = gm.find_rank_locations_of_global_dofs(global_dofs)
                for i, dof in enumerate(global_dofs):
                    cochain = global_cochain[i]
                    rank_positions = local_positions[dof]
                    for position in rank_positions:
                        element, local_numbering = position
                        if element not in new_xi:
                            new_xi[element] = xi[element].copy()
                        else:
                            pass
                        new_xi[element][local_numbering] = cochain
                for e in xi:
                    if e not in new_xi:
                        new_xi[e] = xi[e]
                    else:
                        pass
                x0[ith_unknown] = MseHttStaticLocalVector(new_xi, gm)
            # ------------------------------------------------------------------------
            else:
                pass

        # -- start the Newton iterations ----------------------------------------------------
        xi = x0
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
                        assert linear_term.__class__ is MseHttStaticLocalMatrix, \
                            f"A linear term [{i}][{j}] must be a {MseHttStaticLocalMatrix}."
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

                                assert contribution_2_Aij.__class__ is MseHttStaticLocalMatrix, \
                                    (f"contribution of nonlinear term to linear system must be a "
                                     f"{MseHttStaticLocalMatrix}.")

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

            ls = MseHttStaticLocalLinearSystem(LHS, x, f)

            # adopt customizations: important ------------------------------------------ 2
            for nonlinear_customization in self._nls.customize._nonlinear_customizations:
                indicator = nonlinear_customization['customization_indicator']

                # ------------------------------------------------------------
                if indicator == "fixed_global_dofs_for_unknown":
                    if ITER == 1:
                        assert nonlinear_customization['take-effect'] == 0
                        nonlinear_customization['take-effect'] = 1
                    else:
                        assert nonlinear_customization['take-effect'] == 1
                    ith_unknown = nonlinear_customization['ith_unknown']
                    global_dofs = nonlinear_customization['global_dofs']
                    A = ls.A._A
                    b = ls.b._b
                    Ai_ = A[ith_unknown]
                    bi = b[ith_unknown]
                    for j, Aij in enumerate(Ai_):
                        if ith_unknown != j:
                            Aij.customize.zero_rows(global_dofs)
                        else:
                            Aij.customize.identify_rows(global_dofs)
                    bi.customize.set_values(global_dofs, 0)
                # -------------------------------------------------------------
                else:
                    pass
            # ================================================================================

            als = ls.assemble(preconditioner=preconditioner, threshold=threshold)
            if inner_solver_scheme in ('spsolve', 'direct'):
                solve_x0 = None
            else:
                solve_x0 = 0
            results = als.solve(inner_solver_scheme, x0=solve_x0, **inner_solver_kwargs)

            x, LSm = results[:2]
            ls.x.update(x)
            beta = np.sum(x ** 2) ** 0.5

            BETA.append(beta)
            JUDGE, stop_iteration, convergence_info, JUDGE_explanation = _check_stop_criterion_(
                BETA, atol, ITER, maxiter
            )

            xi1 = list()
            for i, _xi in enumerate(xi):
                dx = ls.x._x[i]
                xi1.append(_xi + dx)

            if stop_iteration:
                break
            else:
                xi = xi1

        # ------ Newton iteration completed xi1 is the solution. --------------------------------------------
        x = xi1
        for k, uk in enumerate(self._nls.unknowns):
            uk.cochain = x[k]  # results sent to the unknowns.
            # Important since yet they are occupied by dx

        t_iteration_end = time()
        Ta = t_iteration_end - t_start
        TiT = t_iteration_end - t_iteration_start

        message += f"<nonlinear_solver>" \
            f" = [RegularNewtonRaphson: {als.solve.A.shape}]" \
            f" of {self._nls._num_nonlinear_terms} nonlinear terms: " \
            f"atol={atol}, maxiter={maxiter} + Linear solver: {inner_solver_scheme} " \
            f"args: {inner_solver_kwargs}" \
            f" -> [ITER: {ITER}]" \
            f" = [beta: %.4e]" \
            f" = [{convergence_info}-{JUDGE_explanation}]" \
            f" -> nLS solving costs %.2f, each ITER cost %.2f" % (beta, Ta, TiT / ITER) \
            + '\n(-*-) Last Linear Solver Message:(-*-)\n' + LSm + '\n'

        info = {
            'total cost': Ta,
            'convergence info': convergence_info,
            'iteration cost': TiT / ITER,
            'residuals': BETA,
        }

        return x, message, info


def _check_stop_criterion_(BETA, atol, ITER, maxiter):
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
        if (not judge_1) and len(BETA) > 1 and BETA[-1] > BETA[-2]:  # error grows by much
            if BETA[-1] > 100 * BETA[-2]:
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
        JUDGE = 0
        stop_iteration = False
        info = None
        JUDGE_explanation = ''

    assert stop_iteration in (True, False), "stop_iteration has to be set."

    return JUDGE, stop_iteration, info, JUDGE_explanation
