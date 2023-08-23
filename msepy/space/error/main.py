# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.space.error.Lambda import MsePySpaceErrorLambda
from msepy.space.error.bundle import MsePySpaceErrorBundle


class MsePySpaceError(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._error = None
        self._freeze()

    def __call__(self, cf, t, local_cochain, degree, quad_degree=None, **kwargs):
        """find the error at time `t`."""
        indicator = self._space.abstract.indicator
        if self._error is None:
            if indicator in ('Lambda', 'bundle-diagonal'):
                self._error = MsePySpaceErrorLambda(self._space)
            elif indicator == 'bundle':
                self._error = MsePySpaceErrorBundle(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._error(cf, t, local_cochain, degree, quad_degree, **kwargs)
