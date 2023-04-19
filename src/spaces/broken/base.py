# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/16/2023 3:19 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from src.tools.frozen import Frozen


class BrokenSpaceBase(Frozen):
    """"""

    def __init__(self, ):
        """"""

        self._freeze()


if __name__ == '__main__':
    # python 
    import __init__ as ph
    print(ph)
