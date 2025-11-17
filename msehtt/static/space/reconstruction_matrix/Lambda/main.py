# -*- coding: utf-8 -*-
"""
"""
from importlib import import_module

from phyem.tools.frozen import Frozen


class MseHttSpaceReconstructMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree, *meshgrid):
        """"""
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        orientation = self._space.orientation
        indicator = f"m{m}n{n}k{k}"
        path = self.__repr__().split('main.')[0][1:] + f"RM_{indicator}"
        module = import_module(path)
        if hasattr(module, 'rm__' + indicator):
            return getattr(module, 'rm__' + indicator)(self._space.tpm, degree, *meshgrid)
        else:
            return getattr(module, 'rm__' + indicator + f"_{orientation}")(
                self._space.tpm, degree, *meshgrid
            )
