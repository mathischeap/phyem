# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.mass_matrix.Lambda import MseHyPy2MassMatrixLambda

from msehy.tools.matrix.dynamic import IrregularDynamicLocalMatrix
from msehy.tools.matrix.static.local import IrregularStaticLocalMatrix


class MseHyPy2MassMatrix(Frozen):
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
        if self._Lambda is None:
            self._Lambda = MseHyPy2MassMatrixLambda(self._space)
        return self._Lambda

    def dynamic(self, degree):
        """Make a IrregularDynamicLocalMatrix for `degree`."""
        M = _StaticMMCaller(self._space, degree)
        return IrregularDynamicLocalMatrix(M)


class _StaticMMCaller(Frozen):
    """"""
    def __init__(self, space, degree):
        """"""
        self._space = space
        self._degree = degree
        self._freeze()

    def __call__(self, *args, g=None, **kwargs):
        """"""
        g = self._generation_caller(g=g)
        degree = self._degree
        M = self._space.mass_matrix(degree, g)
        gm = self._space.gathering_matrix(degree, g)
        M = IrregularStaticLocalMatrix(
            M, gm, gm, cache_key=self._space.mesh[g].metric_signature,
        )
        return M

    def _generation_caller(self, *args, g=None, **kwargs):
        """"""
        _ = args, kwargs
        return self._space._pg(g)
