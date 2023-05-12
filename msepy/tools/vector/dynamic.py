# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:51 PM on 5/12/2023
"""
from tools.frozen import Frozen


class MsePyDynamicLocalVector(Frozen):
    """"""

    def __init__(self, vec_caller):
        """"""
        if callable(vec_caller):
            self._vec_caller = vec_caller
        else:
            raise NotImplementedError()

        self._freeze()

    def __call__(self, *args, **kwargs):
        return self._vec_caller(*args, **kwargs)
