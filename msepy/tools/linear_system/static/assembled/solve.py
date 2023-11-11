# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import linalg as spspalinalg
from tools.frozen import Frozen
from time import time
from tools.miscellaneous.timer import MyTimer
from msepy.form.main import MsePyRootForm
from msepy.tools.matrix.static.assembled import MsePyStaticAssembledMatrix
from msepy.tools.vector.static.assembled import MsePyStaticAssembledVector


class MsePyStaticLinearSystemAssembledSolve(Frozen):
    """"""
    def __init__(self, A, b):
        """"""
        assert A.__class__ is MsePyStaticAssembledMatrix, f"A needs to be a {MsePyStaticAssembledMatrix}"
        assert b.__class__ is MsePyStaticAssembledVector, f"b needs to be a {MsePyStaticAssembledVector}"
        self._A = A
        self._b = b

        self._package = 'scipy'
        self._scheme = 'spsolve'

        self._package_scipy = _PackageScipy()

        self._x0 = None
        self._freeze()

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def package(self):
        return self._package

    @property
    def scheme(self):
        return self._scheme

    @scheme.setter
    def scheme(self, scheme):
        self._scheme = scheme

    @package.setter
    def package(self, package):
        self._package = package

    @property
    def x0(self):
        """the initial guess for iterative solvers. A good initial guess is very important for solving
        large systems.
        """
        return self._x0

    @x0.setter
    def x0(self, _x0):
        """Set the x0. Setting a x0 is very important for large system!"""
        if _x0 == 0:  # make a full zero initial guess

            self._x0 = np.zeros(self.b._gm.num_dofs)

        elif all([_.__class__ is MsePyRootForm for _ in _x0]):   # providing all MsePyRootForm
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

            cochain = np.hstack(cochain)
            assert cochain.shape == self.A._gm_col.shape, f"provided cochain shape wrong!"

            self._x0 = self.A._gm_col.assemble(cochain, mode='replace')

        else:
            raise NotImplementedError()

        assert isinstance(self._x0, np.ndarray), f"x0 must be a ndarray."
        assert self._x0.shape == (self.A._gm_col.num_dofs, ), f"x0 shape wrong!"
        assert self._x0.shape == (self.A.shape[1], ), f"x0 shape wrong!"

    def __call__(self, **kwargs):
        """"""
        assert hasattr(self, f"_package_{self.package}"), f"I have no solver package: {self.package}."
        _package = getattr(self, f"_package_{self.package}")
        assert hasattr(_package, self.scheme), f"package {self.package} has no scheme: {self.scheme}"
        scheme = getattr(_package, self.scheme)

        if self.x0 is None:
            results = scheme(self.A, self.b, **kwargs)
        else:
            results = scheme(self.A, self.b, self.x0, **kwargs)

        return results


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
        # -----------------------------------------------------------------
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

        Compute a vector x such that the 2-norm |b - A x| is minimized.
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
            # --- below we make the preconditioner --------
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

    def gmres(self, A, b, x0, preconditioner=True, **kwargs):
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

        # ----------------------------------------------------------------------------
        A = A._M
        b = b._v
        # ============================================================================

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

    def lgmres(self, A, b, x0, preconditioner=True, **kwargs):
        """"""
        t_start = time()

        # ----------------------------------------------------------------------------
        A = A._M
        b = b._v
        # ============================================================================

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
