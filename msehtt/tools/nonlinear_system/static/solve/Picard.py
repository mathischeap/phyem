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

from msehtt.tools.nonlinear_system.static.solve.Newton_Raphson import _check_stop_criterion_
from msehtt.tools.vector.static.local import concatenate


class MseHtt_NonlinearSystem_Picard(Frozen):
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
            Outer Picard iterations stop when norm(x^{k+1} - x^{k}) < atol.
        maxiter :
            Outer Picard iterations stop when ITER > maxiter.
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

        # initialize the variables ----------------------------------------------------------

        x0 = self.x0
        ITER = 0
        message = ''
        BETA = list()

        # --------------- iterations start --------------------------------------------------
        t_iteration_start = time()

        # -- start the Newton iterations ----------------------------------------------------
        S0, S1 = self._nls.shape
        xi = x0

        while 1:
            ITER += 1
            LHS = [[None for _ in range(S1)] for _ in range(S0)]
            signatures = list()
            for i in range(S0):
                for j in range(S1):
                    linear_term = self._nls._A[i][j]
                    if linear_term is not None:
                        assert linear_term.__class__ is MseHttStaticLocalMatrix, \
                            f"A linear term [{i}][{j}] must be a {MseHttStaticLocalMatrix}."
                        assert LHS[i][j] is None, f"LHS[i][j] must be untouched yet!"
                        signatures.append(linear_term._signature)
                        LHS[i][j] = linear_term
                    else:
                        pass
            signatures = ''.join(signatures)

            rhs = [None for _ in range(S0)]
            f = self._nls._evaluate_nonlinear_terms(xi)
            for i in range(S0):
                rhs[i] = self._nls._b[i]  # b[i] cannot be None
                if f[i] is None:
                    pass
                else:
                    rhs[i] -= MseHttStaticLocalVector(f[i], self._nls._row_gms[i])

            x = list()
            for uk in self._nls.unknowns:
                x.append(uk._f.cochain.static_vec(uk._t))

            ls = MseHttStaticLocalLinearSystem(LHS, x, rhs)

            # adopt customizations: important ---------------------------------------------------
            for nonlinear_customization in self._nls.customize._nonlinear_customizations:
                indicator = nonlinear_customization['customization_indicator']
                signatures += indicator
                # ------------------------------------------------------------
                if indicator == "set_global_dofs_for_unknown":
                    if ITER == 1:
                        assert nonlinear_customization['take-effect'] == 0
                        nonlinear_customization['take-effect'] = 1
                    else:
                        assert nonlinear_customization['take-effect'] == 1
                    ith_unknown = nonlinear_customization['ith_unknown']
                    global_dofs = nonlinear_customization['global_dofs']
                    global_cochain = nonlinear_customization['global_cochain']
                    A = ls.A._A
                    b = ls.b._b
                    Ai_ = A[ith_unknown]
                    bi = b[ith_unknown]
                    for j, Aij in enumerate(Ai_):
                        if ith_unknown != j:
                            Aij.customize.zero_rows(global_dofs)
                        else:
                            Aij.customize.identify_rows(global_dofs)
                    bi.customize.set_values(global_dofs, global_cochain)
                # -------------------------------------------------------------
                else:
                    pass
            # ================================================================================

            if 'unique' in signatures:
                cache = 'unique'
            else:
                signatures = str(self._nls.shape) + '===' + signatures
                raise NotImplementedError(f"Now convert signatures into a cache key! {signatures}")

            als = ls.assemble(cache=cache, preconditioner=preconditioner, threshold=threshold)

            concatenate_xi = concatenate(xi, als.gm_col)
            assembled_xi = concatenate_xi.assemble(vtype='gathered', mode='replace')

            if inner_solver_scheme in ('spsolve', 'direct'):
                solve_x0 = None
            else:
                solve_x0 = assembled_xi

            results = als.solve(inner_solver_scheme, x0=solve_x0, **inner_solver_kwargs)
            x, LSm = results[:2]
            beta = np.sum((assembled_xi.V - x) ** 2) ** 0.5
            BETA.append(beta)
            JUDGE, stop_iteration, convergence_info, JUDGE_explanation = _check_stop_criterion_(
                BETA, atol, ITER, maxiter
            )
            ls.x.update(x)

            if stop_iteration:
                break
            else:
                xi1 = list()
                for i in range(S0):
                    xi1.append(ls.x._x[i])
                xi = xi1

        # ------ Newton iteration completed xi1 is the solution. ---------------------------------

        t_iteration_end = time()
        Ta = t_iteration_end - t_start
        TiT = t_iteration_end - t_iteration_start

        message += f"<nonlinear_solver>" \
            f" = [Regular_Picard: {als.solve.A.shape}]" \
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
