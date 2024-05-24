# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from importlib import import_module


class MseHttSpaceReduceLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, cf_t, degree):
        """"""
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        orientation = self._space.orientation
        indicator = f"m{m}n{n}k{k}"
        path = self.__repr__().split('main.')[0][1:] + f"Rd_{indicator}"
        module = import_module(path)
        if hasattr(module, 'reduce_Lambda__' + indicator):
            return getattr(module, 'reduce_Lambda__' + indicator)(cf_t, self._space.tpm, degree)
        else:
            return getattr(module, 'reduce_Lambda__' + indicator + f"_{orientation}")(
                cf_t, self._space.tpm, degree)
