# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class t3d_ScalarMultiply(Frozen):
    """"""

    def __init__(self, v0, v1):
        """"""
        if isinstance(v0, (int, float)) or isinstance(v1, (int, float)):
            self._number = True

            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                raise Exception
            elif isinstance(v0, (int, float)):
                self._v0_ = v0
                self._v1_ = v1
            elif isinstance(v1, (int, float)):
                self._v0_ = v1
                self._v1_ = v0
            else:
                raise Exception

        else:
            self._number = False
            self._v0_ = v0
            self._v1_ = v1
        self._freeze()

    def __call__(self, t, x, y, z):
        if self._number:
            return self._v0_ * self._v1_(t, x, y, z)
        else:
            return self._v0_(t, x, y, z) * self._v1_(t, x, y, z)
