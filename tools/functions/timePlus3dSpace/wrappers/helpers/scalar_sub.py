# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/13/2022 4:57 PM
"""
from tools.frozen import Frozen


class t3d_ScalarSub(Frozen):
    """"""

    def __init__(self, s0, s1):
        """"""
        self._s0_ = s0
        self._s1_ = s1
        self._freeze()

    def __call__(self, t, x, y, z):
        return self._s0_(t, x, y, z) - self._s1_(t, x, y, z)
