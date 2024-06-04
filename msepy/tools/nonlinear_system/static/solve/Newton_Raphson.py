# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from time import time
from tools.frozen import Frozen
from msepy.form.main import MsePyRootForm
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.tools.linear_system.static.local.main import MsePyStaticLocalLinearSystem


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
            x0, atol=1e-6, maxiter=10,        # args for the outer Newton method.
            # for the inner solver
            inner_solver_package='scipy',
            inner_solver_scheme='spsolve',
            inner_solver_kwargs=None          # args for the inner linear solver.
    ):
        """

        Parameters
        ----------
        x0 :
            The initial guess. Usually be a list of forms. We will parse x0 from the newest cochain of
            these forms.
        atol :
            Outer Newton iterations stop when norm(dx) < atol.
        maxiter :
            Outer Newton iterations stop when ITER > maxiter.
        inner_solver_package :
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

        x0 = self.x0   # a list of 2d arrays;
        assert isinstance(x0, list) and all([isinstance(_, np.ndarray) and _.ndim == 2 for _ in x0]), \
            f"x0 must be a list of 2d array."
        ITER = 0
        message = ''
        BETA = list()

        t_iteration_start = time()

        # ------ some customizations need to be applied here.
        for customization in self._nls.customize._customizations:
            method_name = customization[0]
            method_para = customization[1]

            if method_name == "set_x0_from_local_dofs":
                # we make the #i dof unchanged .....
                i, elements, local_dofs, local_values = method_para
                assert len(elements) == len(local_dofs) == len(local_values), f"positions are not correct."

                x0_i = x0[i]
                assert isinstance(x0_i, np.ndarray) and x0_i.ndim == 2, f'safety check'

                for j,  e in enumerate(elements):
                    x0_i[e, local_dofs[j]] = local_values[j]

                x0[i] = x0_i

            else:
                pass

        xi = x0

        # --- some caches ----------------------------
        _global_dofs_cache = None

        # -- start the Newton iterations ----------------------------
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
                            f"A linear term [{i}][{j}] must be a {MsePyStaticLocalMatrix}."

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
                                     f"{MsePyStaticLocalMatrix}.")

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

            # adopt customizations: important ------------------------------------------ 2
            for customization in self._nls.customize._customizations:
                method_name = customization[0]
                method_para = customization[1]

                A = ls.A._mA  # local matrix A for elements
                f = ls.b._vb  # local vector b for elements
                # ------------------------------------------------------------ 3
                if method_name == "set_no_evaluation":
                    # we make the #i dof unchanged .....
                    dof = method_para
                    A.customize.identify_row(dof)
                    f.customize.set_value(dof, 0)

                elif method_name == 'set_no_evaluation_for_overall_local_dofs':

                    if _global_dofs_cache is None:
                        elements, overall_local_dofs = method_para
                        gm = A._gm0_row
                        global_dofs = gm._find_global_numbering(elements, overall_local_dofs)
                        _global_dofs_cache = global_dofs
                    else:
                        global_dofs = _global_dofs_cache

                    A.customize.identify_diagonal(global_dofs)
                    f.customize.set_values(global_dofs, 0)

                else:
                    pass

            # ==================================================================================

            als = ls.assemble()
            solve = als.solve
            solve.package = inner_solver_package
            solve.scheme = inner_solver_scheme
            if inner_solver_scheme in ('spsolve',):
                pass
            else:
                solve.x0 = 0
            results = solve(**inner_solver_kwargs)
            x, LSm = results[:2]
            ls.x.update(x)
            # results updated to the unknowns of the nonlinear system to make the 2d local cochain

            A_shape = solve._A.shape
            dx = list()
            for f in self._nls.unknowns:
                dx.append(f.cochain.local)

            beta = sum(  # this beta is larger than the real vector norm of dx which can be computed after assembling.
                [
                    np.sum(_**2) for _ in dx
                ]
            )

            BETA.append(beta)
            JUDGE, stop_iteration, convergence_info, JUDGE_explanation = _check_stop_criterion_(
                BETA, atol, ITER, maxiter
            )

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
        x = xi1
        for k, uk in enumerate(self._nls.unknowns):
            uk.cochain = x[k]  # results sent to the unknowns.
            # Important since yet they are occupied by dx

        t_iteration_end = time()
        Ta = t_iteration_end-t_start
        TiT = t_iteration_end-t_iteration_start

        message += f"<nonlinear_solver>" \
            f" = [RegularNewtonRaphson: {A_shape}]" \
            f" of {self._nls._num_nonlinear_terms} nonlinear terms: " \
            f"atol={atol}, maxiter={maxiter} + Linear solver: {inner_solver_package} {inner_solver_scheme} " \
            f"args: {inner_solver_kwargs}" \
            f" -> [ITER: {ITER}]" \
            f" = [beta: %.4e]" \
            f" = [{convergence_info}-{JUDGE_explanation}]" \
            f" -> nLS solving costs %.2f, each ITER cost %.2f" % (BETA[-1], Ta, TiT/ITER) \
            + '\n(-*-) Last Linear Solver Message:(-*-)\n' + LSm + '\n'

        info = {
            'total cost': Ta,
            'convergence info': convergence_info,
            'iteration cost': TiT/ITER,
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
        judge_1 = beta < atol        # judge 1: reach absolute tolerance.
        judge_2 = ITER >= maxiter    # judge 2: reach max iteration number

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
            if progress / beta_old < 0.000001:  # slow converging
                judge_4 = True
            else:
                judge_4 = False
        else:
            judge_4 = False

    # -------------------------------------------------------------------------------
    if any([judge_1, judge_2, judge_3, judge_4]):

        stop_iteration = True

        if judge_1:    # reach atol
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
