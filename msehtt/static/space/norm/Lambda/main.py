# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from importlib import import_module


class MseHttSpaceNormLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree, cochain, norm_type='L2'):
        """"""
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        orientation = self._space.orientation
        indicator = f"m{m}n{n}k{k}"
        path = self.__repr__().split('main.')[0][1:] + f"norm_{indicator}"
        module = import_module(path)
        if hasattr(module, 'norm_Lambda__' + indicator):
            return getattr(module, 'norm_Lambda__' + indicator)(
                self._space.tpm, degree, cochain, norm_type=norm_type,
            )
        else:
            return getattr(module, 'norm_Lambda__' + indicator + f"_{orientation}")(
                self._space.tpm, degree, cochain, norm_type=norm_type
            )
