# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.basis_functions.Lambda import MseHyPy2BasisFunctionsLambda


class MseHyPy2BasisFunctions(Frozen):
    """"""

    def __init__(self, space):
        """Generation in-dependent."""
        self._space = space
        self._Lambda_cache = {}
        self._freeze()

    def __getitem__(self, degree):
        """Generation in-dependent."""
        indicator = self._space.abstract.indicator
        key = str(degree)
        if indicator in ('Lambda', ):
            if key in self._Lambda_cache:
                Lambda_bf = self._Lambda_cache[key]
            else:
                Lambda_bf = MseHyPy2BasisFunctionsLambda(self._space, degree)
                self._Lambda_cache[key] = Lambda_bf
            return Lambda_bf

        else:
            raise NotImplementedError()
