# -*- coding: utf-8 -*-
r"""
"""
from random import random

from phyem.tools.frozen import Frozen


class TimeSpaceFunctionBase(Frozen):
    """"""

    def __init__(self, steady, allowed_time_range=None):
        r"""

        Parameters
        ----------
        steady :
            If it is steady, then this vector is independent of t.

        allowed_time_range :
            to evaluate this scalar, we can only use t in this range.

        """
        self.___is_steady___ = steady
        self.allowed_time_range = allowed_time_range

    @staticmethod
    def _is_time_space_func():
        """"""
        return True

    @property
    def allowed_time_range(self):
        r"""allowed_time_range"""
        return self._allowed_time_range_

    @allowed_time_range.setter
    def allowed_time_range(self, atr):
        r""""""
        if atr is None:
            pass
        else:
            raise NotImplementedError()
        self._allowed_time_range_ = atr

    def atr_check(self, t):
        r"""`allowed-time-range` check.

        We check whether the input `t` is allowed by the atr of this function.
        """
        if self.allowed_time_range is None:
            return True
        else:
            raise NotImplementedError(t)

    def _find_random_testing_time_instance_(self):
        r""""""
        if self.allowed_time_range is None:
            return random()
        else:
            raise NotImplementedError()
