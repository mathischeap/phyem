# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import linalg as spspalinalg
from tools.frozen import Frozen
from time import time
from tools.miscellaneous.timer import MyTimer
from msepy.form.main import MsePyRootForm


class MsePyStaticLinearSystemAssembledSolve(Frozen):
    """"""
    def __init__(self, als):
        """"""
        self._als = als
        self._A = als.A._M
        self._b = als.b._v
        self._system_info = {
            'shape': self._A.shape
        }
        self._message = ''
        self._info = None

        self._package = 'scipy'
        self._scheme = 'spsolve'

        # implemented packages
        self._package_scipy = _PackageScipy(self, self._system_info)
        self._package_mkl = _PackageMKL(self, self._system_info)

        self._x0 = None
        self._freeze()

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
        if self._x0 is None:
            raise Exception(f"x0 is None, first set it.")
        return self._x0

    @x0.setter
    def x0(self, _x0):
        """Set the x0. Setting a x0 is very important for large system!"""
        if _x0 == 0:  # make a full zero initial guess

            self._x0 = np.zeros(self._als.b._gm.num_dofs)

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
            assert cochain.shape == self._als.b._gm.shape, f"provided cochain shape wrong!"

            self._x0 = self._als.b._gm.assemble(cochain, mode='replace')

        else:
            raise NotImplementedError()

        assert isinstance(self._x0, np.ndarray), f"x0 must be a ndarray."
        assert self._x0.shape == (self._als.b._gm.num_dofs, ), f"x0 shape wrong!"
        assert self._x0.shape == (self._A.shape[1], ), f"x0 shape wrong!"

    @property
    def message(self):
        """return the message of the last solver."""
        return self._message

    @property
    def info(self):
        """store the info of the last solver."""
        return self._info

    def __call__(self, update_x=True, **kwargs):
        """"""
        assert hasattr(self, f"_package_{self.package}"), f"I have no solver package: {self.package}."
        _package = getattr(self, f"_package_{self.package}")
        assert hasattr(_package, self.scheme), f"package {self.package} has no scheme: {self.scheme}"
        results = getattr(_package, self.scheme)(self._A, self._b, **kwargs)
        if update_x:
            self._als._static.x.update(results[0])
        else:
            pass
        self._message = results[1]
        self._info = results[2]
        return results


class _PackageMKL(Frozen):
    """"""
    def __init__(self, solve, system_info):
        self._solve = solve
        self._system_info = system_info
        self._freeze()

    def qr(self, A, b, **kwargs):
        """"""
        from sparse_dot_mkl import sparse_qr_solve_mkl

        t_start = time()
        x = sparse_qr_solve_mkl(A, b, **kwargs)
        t_cost = time() - t_start
        t_cost = MyTimer.seconds2dhms(t_cost)
        message = f"Linear system of shape: {self._system_info['shape']}" + \
                  f" <sparse_qr_solve_mkl costs: {t_cost}> "
        info = {
            'total cost': t_cost,
        }
        return x, message, info


class _PackageScipy(Frozen):
    """"""
    def __init__(self, solve, system_info):
        self._solve = solve
        self._system_info = system_info
        self._freeze()

    def spsolve(self, A, b, **kwargs):
        """direct solver."""
        if len(kwargs) > 0:
            print(f'warning: kwargs={kwargs} have no affects on scipy spsolve.')
        else:
            pass

        t_start = time()
        x = spspalinalg.spsolve(A, b)
        t_cost = time() - t_start
        t_cost = MyTimer.seconds2dhms(t_cost)
        message = f"Linear system of shape: {self._system_info['shape']}" + \
                  f" <direct solver costs: {t_cost}> "
        info = {
            'total cost': t_cost,
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

    def gmres(self, A, b, preconditioner=True, **kwargs):
        """

        Parameters
        ----------
        A
        b
        preconditioner :
            None, True, or use a particular preconditioner by, for example,
                preconditioner = {
                    'name': p_name,
                    'para1_name': p1,
                    ...,
                }
        kwargs

        Returns
        -------

        """
        t_start = time()

        preconditioner = self._parse_preconditioner(A, preconditioner)

        if self._solve._x0 is None:  # bad for large system. Better providing x0
            x, info = spspalinalg.gmres(
                A, b, M=preconditioner, **kwargs  # by default, x0=None. spspalinalg.gmres will make a x0 then.
            )
        else:
            x, info = spspalinalg.gmres(
                A, b, x0=self._solve.x0, M=preconditioner, **kwargs
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
        message = f"Linear system of shape: {self._system_info['shape']} " + \
                  f"<gmres costs: {t_cost}> <info: {info}> <inputs: {info_kwargs}>"
        info = {
            'total cost': t_cost,
            'convergence info': info,
        }
        return x, message, info

    def lgmres(self, A, b, preconditioner=True, **kwargs):
        """"""
        t_start = time()

        preconditioner = self._parse_preconditioner(A, preconditioner)

        if self._solve._x0 is None:  # bad for large system. Better providing x0
            x, info = spspalinalg.lgmres(
                A, b, M=preconditioner, **kwargs
            )
        else:
            x, info = spspalinalg.lgmres(
                A, b, x0=self._solve.x0, M=preconditioner, **kwargs
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
        message = f"Linear system of shape: {self._system_info['shape']} " + \
                  f"<lgmres costs: {t_cost}> <info: {info}> <inputs: {info_kwargs}>"
        info = {
            'total cost': t_cost,
            'convergence info': info,
        }
        return x, message, info
