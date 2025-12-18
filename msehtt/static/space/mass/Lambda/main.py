# -*- coding: utf-8 -*-
r"""
"""
from importlib import import_module

from phyem.tools.frozen import Frozen


class MseHttSpace_MASS_Lambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree, cochain):
        """"""
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        indicator = f"m{m}n{n}k{k}"
        path = self.__repr__().split('main.')[0][1:] + f"mass_{indicator}"
        module = import_module(path)
        return getattr(module, 'mass_Lambda__' + indicator)(
            self._space.tpm, degree, cochain,
        )
