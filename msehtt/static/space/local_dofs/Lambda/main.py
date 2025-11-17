# -*- coding: utf-8 -*-
r"""
"""
from importlib import import_module

from phyem.tools.frozen import Frozen


class MseHttSpace_Local_Dofs_Lambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree):
        """"""
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        orientation = self._space.orientation
        indicator = f"m{m}n{n}k{k}"
        path = self.__repr__().split('main.')[0][1:] + f"LDofs_{indicator}"
        module = import_module(path)
        if hasattr(module, 'LDofs_Lambda__' + indicator):
            return getattr(module, 'LDofs_Lambda__' + indicator)(
                self._space.tpm, degree,
            )
        else:
            return getattr(module, 'LDofs_Lambda__' + indicator + f"_{orientation}")(
                self._space.tpm, degree,
            )
