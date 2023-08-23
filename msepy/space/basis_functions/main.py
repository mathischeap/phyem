# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.space.basis_functions.Lambda import MsePyBasisFunctionsLambda
from msepy.space.basis_functions.bundle import MsePyBasisFunctionsBundle


class MsePyBasisFunctions(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda_cache = {}
        self._bundle_cache = {}
        self._bundle_diagonal_cache = {}
        self._freeze()

    def __getitem__(self, degree):
        """Return"""
        indicator = self._space.abstract.indicator
        key = str(degree)
        if indicator in ('Lambda', 'bundle-diagonal'):
            if key in self._Lambda_cache:
                Lambda_bf = self._Lambda_cache[key]
            else:
                Lambda_bf = MsePyBasisFunctionsLambda(self._space, degree)
                self._Lambda_cache[key] = Lambda_bf
            return Lambda_bf

        elif indicator == 'bundle':
            if key in self._bundle_cache:
                bundle_bf = self._bundle_cache[key]
            else:
                bundle_bf = MsePyBasisFunctionsBundle(self._space, degree)
                self._bundle_cache[key] = bundle_bf
            return bundle_bf

        else:
            raise NotImplementedError()
