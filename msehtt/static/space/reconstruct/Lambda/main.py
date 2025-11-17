# -*- coding: utf-8 -*-
"""
"""
from importlib import import_module

from phyem.tools.frozen import Frozen


class MseHttSpaceReconstructLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree, cochain, *meshgrid, ravel=False, element_range=None):
        """"""
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        orientation = self._space.orientation
        indicator = f"m{m}n{n}k{k}"
        path = self.__repr__().split('main.')[0][1:] + f"Rc_{indicator}"
        module = import_module(path)
        if hasattr(module, 'reconstruct_Lambda__' + indicator):
            return getattr(module, 'reconstruct_Lambda__' + indicator)(
                self._space.tpm, degree, cochain, *meshgrid, ravel=ravel, element_range=element_range,
            )
        else:
            return getattr(module, 'reconstruct_Lambda__' + indicator + f"_{orientation}")(
                self._space.tpm, degree, cochain, *meshgrid, ravel=ravel, element_range=element_range,
            )
