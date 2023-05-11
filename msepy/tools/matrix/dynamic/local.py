# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:56 PM on 5/11/2023
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix


class MsePyDynamicLocalMatrix(Frozen):
    """"""

    def __init__(self, data):
        """"""
        if data.__class__ is MsePyStaticLocalMatrix:
            self._dtype = 'static'
            self._static_data = data
        elif isinstance(data, list):
            self._dtype = 'list'
            self._list_data = data
        elif callable(data):
            self._dtype = 'dynamic'
            self._callable_data = data
        else:
            raise NotImplementedError(f"MsePyDynamicLocalMatrix cannot take {data}.")

        self._freeze()

    def __call__(self, *args, **kwargs):
        """Gives a static local matrix by evaluating the dynamic local matrix with `*args, **kwargs`."""
        if self._dtype == 'static':
            return self._static_data
        elif self._dtype == 'list':
            # noinspection PyUnresolvedReferences
            mat = self._list_data[0](*args, **kwargs)
            for data in self._list_data[1:]:
                mat = mat @ data(*args, **kwargs)
            return mat
        elif self._dtype == 'dynamic':
            return self._callable_data(*args, **kwargs)
        else:
            raise NotImplementedError(f"data type = {self._dtype} is wrong!")

    def __matmul__(self, other):
        """self @ other"""
        assert other.__class__ is MsePyDynamicLocalMatrix, f"MsePyDynamicLocalMatrix cannot @ {other.__class__}."

        mat_list = [self, other]

        return MsePyDynamicLocalMatrix(mat_list)


if __name__ == '__main__':
    # python 
    pass
