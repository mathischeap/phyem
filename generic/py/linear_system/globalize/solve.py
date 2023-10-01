# -*- coding: utf-8 -*-
r"""
"""
from scipy.sparse import linalg as spspalinalg
from tools.frozen import Frozen
from time import time
from tools.miscellaneous.timer import MyTimer


class Solve(Frozen):
    """"""
    def __init__(self, A, b):
        """"""
        self._A = A
        self._b = b
        self._package = 'scipy'
        self._scheme = 'spsolve'

        # implemented packages
        self._package_scipy = _PackageScipy()

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

    def __call__(self, **kwargs):
        """"""
        assert hasattr(self, f"_package_{self.package}"), f"I have no solver package: {self.package}."
        _package = getattr(self, f"_package_{self.package}")
        assert hasattr(_package, self.scheme), f"package {self.package} has no scheme: {self.scheme}"
        results = getattr(_package, self.scheme)(self._A, self._b, **kwargs)
        return results


class _PackageScipy(Frozen):
    """"""
    def __init__(self):
        self._freeze()

    @staticmethod
    def spsolve(A, b):
        """direct solver."""
        t_start = time()
        x = spspalinalg.spsolve(A, b)
        t_cost = time() - t_start
        t_cost = MyTimer.seconds2dhms(t_cost)
        message = f"Linear system of shape: {A.shape}" + \
                  f" <direct solver costs: {t_cost}> "
        info = {
            'total cost': t_cost,
        }
        return x, message, info
