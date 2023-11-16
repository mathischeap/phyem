# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msepy.form.main import MsePyRootForm
from msepy.tools.matrix.static.assembled import MsePyStaticAssembledMatrix
from msepy.tools.vector.static.assembled import MsePyStaticAssembledVector

from msepy.tools.linear_system.static.assembled.solvers._scipy import _PackageScipy


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
