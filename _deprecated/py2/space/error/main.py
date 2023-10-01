# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.error.Lambda import MseHyPy2SpaceErrorLambda


class MseHyPy2SpaceError(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._error = None
        self._freeze()

    def __call__(self, cf, cochain, quad_degree=None, **kwargs):
        """find the error at time `t`."""
        indicator = self._space.abstract.indicator
        if self._error is None:
            if indicator in ('Lambda', ):
                self._error = MseHyPy2SpaceErrorLambda(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._error(cf, cochain, quad_degree, **kwargs)
