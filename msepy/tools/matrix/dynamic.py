# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:56 PM on 5/11/2023
"""
from tools.frozen import Frozen
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.form.cochain.vector.dynamic import MsePyRootFormDynamicCochainVector
from msepy.tools.vector.dynamic import MsePyDynamicLocalVector


class MsePyDynamicLocalMatrixVector(Frozen):
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
            static = self._static_data
        elif self._dtype == 'list':
            # noinspection PyUnresolvedReferences
            mat = self._list_data[0](*args, **kwargs)
            for data in self._list_data[1:]:
                mat = mat @ data(*args, **kwargs)
            static = mat
        elif self._dtype == 'dynamic':
            static = self._callable_data(*args, **kwargs)
        else:
            raise NotImplementedError(f"data type = {self._dtype} is wrong!")

        assert isinstance(static, MsePyStaticLocalMatrix), f"must return a static one!"

        return static

    def __matmul__(self, other):
        """self @ other"""
        if other.__class__ is MsePyDynamicLocalMatrixVector:

            mat_list = [self, other]

            return MsePyDynamicLocalMatrixVector(mat_list)

        elif other.__class__ in (MsePyDynamicLocalVector, MsePyRootFormDynamicCochainVector):

            _matmul = _MatMulDynamicLocalVector(self, other)

            return MsePyDynamicLocalVector(_matmul)

        else:
            raise NotImplementedError(f"{self.__class__.__name__} cannot @ {other.__class__.__name__}")


class _MatMulDynamicLocalVector(Frozen):
    """"""

    def __init__(self, dyMat, dyVec):
        self._mat = dyMat
        self._vec = dyVec
        self._freeze()

    def __call__(self, *args, **kwargs):

        static_mat = self._mat(*args, **kwargs)
        static_vec = self._vec(*args, **kwargs)

        return static_mat @ static_vec
