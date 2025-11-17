# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msepy.space.norm.Lambda import MsePySpaceNormLambda
from phyem.msepy.space.norm.bundle import MsePySpaceNormBundle


class MsePySpaceNorm(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._norm = None
        self._freeze()

    def __call__(self, local_cochain, degree, quad_degree=None, **kwargs):
        """find the error at time `t`."""
        indicator = self._space.abstract.indicator
        if self._norm is None:
            if indicator == 'Lambda':
                self._norm = MsePySpaceNormLambda(self._space)
            elif indicator == 'bundle':
                self._norm = MsePySpaceNormBundle(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._norm(local_cochain, degree, quad_degree, **kwargs)
