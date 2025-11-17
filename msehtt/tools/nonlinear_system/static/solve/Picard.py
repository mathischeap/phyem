# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from time import time

from phyem.tools.frozen import Frozen
from phyem.msehtt.static.form.main import MseHttForm
from phyem.msehtt.tools.vector.static.local import MseHttStaticLocalVector
from phyem.msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from phyem.msehtt.tools.linear_system.static.local.main import MseHttStaticLocalLinearSystem

from phyem.msehtt.tools.nonlinear_system.static.solve.Newton_Raphson import _check_stop_criterion_
from phyem.msehtt.tools.vector.static.local import concatenate

from phyem.src.config import COMM, MPI


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
                elif isinstance(f[i], MseHttStaticLocalVector):
                    assert f[i]._gm == self._nls._row_gms[i], f"must be"
                    rhs[i] -= f[i]
                else:
                    rhs[i] -= MseHttStaticLocalVector(f[i], self._nls._row_gms[i])

            x = list()
            for uk in self._nls.unknowns:
                x.append(uk._f.cochain.static_vec(uk._t))

            ls = MseHttStaticLocalLinearSystem(LHS, x, rhs)

            customizations_to_be_handled_by_LinearSystem_Assembler = list()
            select_values_of_res_x_rule_keys = list()
            # a list of values that define how to clean x before sending it to unknowns

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
                elif indicator == "add_additional_constrain__fix_a_global_dof":
                    if ITER == 1:
                        assert nonlinear_customization['take-effect'] == 0
                        nonlinear_customization['take-effect'] = 1
                    else:
                        assert nonlinear_customization['take-effect'] == 1

                    ith_unknown = nonlinear_customization['ith_unknown']
                    global_dof = nonlinear_customization['global_dof']
                    insert_place = nonlinear_customization['insert_place']

                    the_unknown = xi[ith_unknown]
                    the_gm = the_unknown._gm
                    e, local_dof = the_gm.find_representative_location(global_dof)
                    if e in the_unknown:
                        initial_guess = the_unknown[e][global_dof]
                    else:
                        initial_guess = 0
                    initial_guess = COMM.allreduce(initial_guess, op=MPI.SUM)

                    if insert_place == -1:
                        customizations_to_be_handled_by_LinearSystem_Assembler.append(
                            {
                                'A': ['new_EndZeroRowCol_with_a_one_for_global_dof', ith_unknown, global_dof],
                                'b': ['add_a_value_at_the_end', initial_guess],
                            }
                        )
                        select_values_of_res_x_rule_keys.append("InsertPlaceEnd")
                    else:
                        raise NotImplementedError()

                # -------------------------------------------------------------
                else:
                    pass
            # ================================================================================

            if 'unique' in signatures:
                cache = 'unique'
            else:
                signatures = str(self._nls.shape) + '===' + signatures
                raise NotImplementedError(f"Now convert signatures into a cache key! {signatures}")

            als = ls.assemble(
                cache=cache,
                preconditioner=preconditioner,
                threshold=threshold,
                customizations=customizations_to_be_handled_by_LinearSystem_Assembler
            )

            if als.gm_col is None:
                gm_COL = ls.A._mA._gm_col
            else:
                gm_COL = als.gm_col

            concatenate_xi = concatenate(xi, gm_COL)
            assembled_xi = concatenate_xi.assemble(vtype='gathered', mode='replace')

            if inner_solver_scheme in ('spsolve', 'direct'):
                solve_x0 = None
            else:
                solve_x0 = assembled_xi

            results = als.solve(inner_solver_scheme, x0=solve_x0, **inner_solver_kwargs)
            x, LSm = results[:2]

            if not select_values_of_res_x_rule_keys:  # when it is empty
                pass

            elif len(select_values_of_res_x_rule_keys) == 1:
                # we have received a list of a single value that says how we clean x.
                if select_values_of_res_x_rule_keys[0] == 'InsertPlaceEnd':
                    # we have received a list: ['InsertPlaceEnd', ], this means we just need to drop the last value.
                    # So we just drop the last value of x.
                    x = x[:-1]
                else:
                    raise NotImplementedError(select_values_of_res_x_rule_keys)
            else:
                raise NotImplementedError(
                    f"For select_values_of_res_x_rule_keys={select_values_of_res_x_rule_keys}, not coded yet."
                )

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
