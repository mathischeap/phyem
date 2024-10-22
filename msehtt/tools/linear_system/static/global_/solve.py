# -*- coding: utf-8 -*-
r"""
"""
import sys
import numpy as np

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msehtt.tools.vector.static.local import MseHttStaticLocalVector
from msehtt.tools.vector.static.local import concatenate
from msehtt.tools.vector.static.global_gathered import MseHttGlobalVectorGathered
import msehtt.tools.linear_system.static.global_.solvers.mpi_py as _mpi_py
import msehtt.tools.linear_system.static.global_.solvers.scipy_ as _scipy
import msehtt.tools.linear_system.static.global_.solvers.pypardiso_ as _pypardiso

from msehtt.static.form.main import MseHttForm


___local_setting___ = {
    'safe first try': 7,
}


class MseHttLinearSystemSolve(Frozen):
    r""""""

    def __init__(self, Axb):
        """Axb is the linear system. So in order to use this class, we must start with a linear system."""
        self._Axb = Axb
        self._package_mpi_py = _mpi_py
        self._package_scipy = _scipy
        self._package_pypardiso = _pypardiso
        self._freeze()

    @property
    def A(self):
        """`A` of the linear system `Ax=b`."""
        return self._Axb.A

    @property
    def b(self):
        """`b` of the linear system `Ax=b`."""
        return self._Axb.b

    def __call__(self, scheme, x0=None, package=None, **kwargs):
        """"""
        package_name, scheme_name = self._package_scheme_parser_(package, scheme)
        x0 = self._x0_parser_(x0)
        assert hasattr(self, f"_package_{package_name}"), f"I have no solver package: {package_name}."
        _package = getattr(self, f"_package_{package_name}")
        assert hasattr(_package, scheme_name), f"package {scheme_name} has no scheme: {scheme_name}"
        scheme = getattr(_package, scheme_name)
        if x0 is None:
            results = scheme(self.A, self.b, **kwargs)
        else:
            results = scheme(self.A, self.b, x0, **kwargs)
        return results

    def _package_scheme_parser_(self, package_name, scheme_name):
        """"""
        if package_name is None:  # provide scheme_indicator
            if scheme_name in ('spsolve', 'direct'):
                package_name = 'scipy'
                scheme_name = 'spsolve'
            elif scheme_name == 'ppsp':
                package_name = 'pypardiso'
                scheme_name = 'spsolve'
            elif scheme_name == 'gmres':
                package_name = 'mpi_py'
            elif scheme_name == 'lgmres':
                package_name = 'mpi_py'
            else:
                raise NotImplementedError(f"default package not set for scheme={scheme_name}, set it manually.")
        else:
            pass

        return package_name, scheme_name

    def _x0_parser_(self, x0):
        """"""
        if x0 is None:
            return None
        else:
            pass

        if x0.__class__ is MseHttGlobalVectorGathered:
            pass

        else:

            if x0 == 0:  # the initial guess is a zero-vector.
                shape = self._Axb.shape
                V = np.zeros(shape[1])
                x0 = MseHttGlobalVectorGathered(V, gm=self._Axb.gm_col)

            else:
                if not isinstance(x0, (list, tuple)):
                    x0 = [x0, ]
                else:
                    pass

                gms = list()
                X0_ = list()

                for x0i in x0:
                    if x0i.__class__ is MseHttForm:
                        # we receive a msehtt root form: we use its newest cochain to replace its place in x0.
                        gms.append(x0i.cochain.gathering_matrix)
                        if x0i.cochain.newest is None:  # if it has no cochain yet. We make it all zero.
                            vec = MseHttStaticLocalVector(0, x0i.cochain.gathering_matrix)
                        else:  # otherwise, we use its newest cochain.
                            vec = x0i.cochain.static_vec(x0i.cochain.newest)
                        X0_.append(vec)
                    else:
                        raise NotImplementedError()

                x0 = concatenate(X0_, gms).assemble(vtype='gathered', mode='replace')

        sft = ___local_setting___['safe first try']
        if sft > 0:
            essential_dofs, essential_values = self.A.___find_essential_dof_coefficients___(self.b)
            x0._V[essential_dofs] = essential_values
            ___local_setting___['safe first try'] = sft - 1

        assert x0.__class__ is MseHttGlobalVectorGathered, f"x0 type wrong."
        assert x0.shape == (self._Axb.shape[1],),  f"x0 shape wrong."
        return x0
