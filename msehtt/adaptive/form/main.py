# -*- coding: utf-8 -*-
r"""
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen


class ClassName(Frozen):
    """"""

    def __init__(self):
        """"""
        self._freeze()


if __name__ == '__main__':
    # mpiexec -n 4 python 
    pass
