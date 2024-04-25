# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from importlib import import_module


class MseHttSpaceLocalNumberingLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, etype, degree):
        """"""
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        orientation = self._space.orientation
        indicator = f"m{m}n{n}k{k}"
        path = self.__repr__().split('main.')[0][1:] + f"_{indicator}_"
        module = import_module(path)
        if hasattr(module, 'local_numbering_Lambda__' + indicator):
            return getattr(module, 'local_numbering_Lambda__' + indicator)(etype, degree)
        else:
            return getattr(module, 'local_numbering_Lambda__' + indicator + f"_{orientation}")(etype, degree)
