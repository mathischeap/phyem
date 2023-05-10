# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 11:10 AM on 5/10/2023
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen


class MsePyRootFormDynamicCopy(Frozen):
    """"""

    def __init__(self, rf, t_func):
        """"""
        self._f = rf
        self._tf = t_func  # a function to be valued; returns a time instant which then leads a particular static copy.
        self._freeze()


if __name__ == '__main__':
    # python msepy/form/dynamic.py
    pass
