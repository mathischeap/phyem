# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehtt.static.space.gathering_matrix.Lambda.main import MseHttSpaceGatheringMatrixLambda
from importlib import import_module


class MseHttSpaceGatheringMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree):
        """"""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceGatheringMatrixLambda(self._space)(degree)
        else:
            raise NotImplementedError()

    def _next(self, degree):
        """"""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            m = self._space.m
            n = self._space.n
            k = self._space.abstract.k
            orientation = self._space.orientation
            path = self.__repr__().split('main.')[0][1:]

            self_indicator = f"m{m}n{n}k{k}_{orientation}"
            if self_indicator == "m2n2k0_inner":
                path += f"Lambda.GM_m2n2k1"
                module = import_module(path)
                return getattr(module, 'gathering_matrix_Lambda__m2n2k1_inner')(self._space.tpm, degree)
            elif self_indicator == "m2n2k0_outer":
                path += f"Lambda.GM_m2n2k1"
                module = import_module(path)
                return getattr(module, 'gathering_matrix_Lambda__m2n2k1_outer')(self._space.tpm, degree)
            elif self_indicator == "m2n2k1_inner":
                path += f"Lambda.GM_m2n2k2"
                module = import_module(path)
                return getattr(module, 'gathering_matrix_Lambda__m2n2k2')(self._space.tpm, degree)
            elif self_indicator == "m2n2k1_outer":
                path += f"Lambda.GM_m2n2k2"
                module = import_module(path)
                return getattr(module, 'gathering_matrix_Lambda__m2n2k2')(self._space.tpm, degree)
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()
