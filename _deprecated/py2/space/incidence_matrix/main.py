# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.incidence_matrix.Lambda import MseHyPy2IncidenceMatrixLambda

from msehy.tools.matrix.dynamic import IrregularDynamicLocalMatrix
from msehy.tools.matrix.static.local import IrregularStaticLocalMatrix


class MseHyPy2IncidenceMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._freeze()

    def __call__(self, degree, g):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree, g)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        """"""
        if self._Lambda is None:
            self._Lambda = MseHyPy2IncidenceMatrixLambda(self._space)
        return self._Lambda

    def dynamic(self, degree):
        """"""
        M = _StaticECaller(self._space, degree)
        return IrregularDynamicLocalMatrix(M)


class _StaticECaller(Frozen):
    """"""
    def __init__(self, space, degree):
        """"""
        self._space = space
        self._degree = degree
        self._freeze()

    def __call__(self, *args, g=None, **kwargs):
        """"""
        _ = args, kwargs
        g = self._generation_caller(g=g)
        degree = self._degree
        gm1_col = self._space.gathering_matrix(degree, g)
        gm0_row = self._space.gathering_matrix._next(degree, g)

        E = self._space.incidence_matrix(degree, g)

        E = IrregularStaticLocalMatrix(
            E, gm0_row, gm1_col, cache_key=self._space.mesh[g].fc_type_signature,
        )
        return E

    def _generation_caller(self, *args, g=None, **kwargs):
        """"""
        _ = args, kwargs
        return self._space._pg(g)
