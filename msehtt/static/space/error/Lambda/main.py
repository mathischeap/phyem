# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from importlib import import_module


class MseHttSpaceLambdaError(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, cf, cochain, degree, error_type):
        """"""
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        orientation = self._space.orientation
        indicator = f"m{m}n{n}k{k}"
        path = self.__repr__().split('main.')[0][1:] + f"Er_{indicator}"
        module = import_module(path)
        if hasattr(module, 'error__' + indicator):
            return getattr(module, 'error__' + indicator)(
                self._space.tpm, cf, cochain, degree, error_type)
        else:
            return getattr(module, 'error__' + indicator + f"_{orientation}")(
                self._space.tpm, cf, cochain, degree, error_type)
