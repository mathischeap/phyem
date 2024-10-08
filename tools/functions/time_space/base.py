# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class TimeSpaceFunctionBase(Frozen):
    """"""

    def __init__(self, steady):
        self.___is_steady___ = steady

    @staticmethod
    def _is_time_space_func():
        """"""
        return True
