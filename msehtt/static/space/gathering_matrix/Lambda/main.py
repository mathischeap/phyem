# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from importlib import import_module


class MseHttSpaceGatheringMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, tpm, degree):
        """"""
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        orientation = self._space.orientation
        indicator = f"m{m}n{n}k{k}"
        path = self.__repr__().split('main.')[0][1:] + f"GM_{indicator}"
        module = import_module(path)
        if hasattr(module, 'gathering_matrix_Lambda__' + indicator):
            return getattr(module, 'gathering_matrix_Lambda__' + indicator)(tpm, degree)
        else:
            return getattr(module, 'gathering_matrix_Lambda__' + indicator + f"_{orientation}")(tpm, degree)
