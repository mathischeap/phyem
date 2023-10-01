# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.tools.matrix.static.local import IrregularStaticLocalMatrix
from msehy.tools.vector.dynamic import IrregularDynamicLocalVector, IrregularDynamicCochainVector


class IrregularDynamicLocalMatrix(Frozen):
    """"""

    def __init__(self, data):
        """"""
        # if data.__class__ is IrregularStaticLocalMatrix:
        #     self._dtype = 'static'
        #     self._static_data = data
        if isinstance(data, list):
            self._dtype = 'list'
            self._list_data = data
        elif callable(data):
            self._dtype = 'dynamic'
            self._callable_data = data
        else:
            raise NotImplementedError(f"MsePyDynamicLocalMatrix cannot take {data}.")

        self._freeze()

    def __call__(self, *args, g=None, **kwargs):
        """Gives a static local matrix by evaluating the dynamic local matrix with `*args, **kwargs`."""
        # if self._dtype == 'static':
        #     static = self._static_data
        if self._dtype == 'list':
            # noinspection PyUnresolvedReferences
            mat = self._list_data[0](*args, g=g, **kwargs)
            for data in self._list_data[1:]:
                mat = mat @ data(*args, g=g, **kwargs)
            static = mat
        elif self._dtype == 'dynamic':
            static = self._callable_data(*args, g=g, **kwargs)
        else:
            raise NotImplementedError(f"data type = {self._dtype} is wrong!")

        assert isinstance(static, IrregularStaticLocalMatrix), f"must return a static one!"

        return static

    def __matmul__(self, other):
        """self @ other"""
        if other.__class__ is IrregularDynamicLocalMatrix:

            mat_list = [self, other]

            return IrregularDynamicLocalMatrix(mat_list)

        elif other.__class__ in (IrregularDynamicLocalVector, IrregularDynamicCochainVector):

            _matmul = _MatMulDynamicLocalVector(self, other)

            return IrregularDynamicLocalVector(_matmul)

        else:
            raise NotImplementedError(f"{self.__class__.__name__} cannot @ {other.__class__.__name__}")

    @property
    def T(self):
        """"""
        dynamic_T_matrix = _DynamicTransposeMatrix(self)
        return self.__class__(dynamic_T_matrix)


class _DynamicTransposeMatrix(Frozen):
    """"""
    def __init__(self, M):
        """"""
        assert M._dtype == 'dynamic'
        self._M = M
        self._freeze()

    def __call__(self, *args, g=None, **kwargs):
        """"""
        return self._M(*args, g=g, **kwargs).T


class _MatMulDynamicLocalVector(Frozen):
    """"""

    def __init__(self, dyMat, dyVec):
        """"""
        self._mat = dyMat
        self._vec = dyVec
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        static_mat = self._mat(*args, **kwargs)
        static_vec = self._vec(*args, **kwargs)

        return static_mat @ static_vec
