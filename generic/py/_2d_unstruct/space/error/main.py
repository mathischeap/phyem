# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from generic.py._2d_unstruct.space.error.Lambda import ErrorLambda


class Error(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._error = None
        self._freeze()

    def __call__(self, cf, cochain, d=2):
        """find the error at time `t`."""
        indicator = self._space.abstract.indicator
        if self._error is None:
            if indicator in ('Lambda', ):
                self._error = ErrorLambda(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._error(cf, cochain, d=2)
