# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from phyem.msehtt.static.form.cochain.vector.dynamic import MseHttDynamicCochainVector
from phyem.msehtt.tools.vector.dynamic import MseHttDynamicLocalVector


class MseHttDynamicLocalMatrix(Frozen):
    """"""

    def __init__(self, data):
        """"""
        if data.__class__ is MseHttStaticLocalMatrix:
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

        assert isinstance(static, MseHttStaticLocalMatrix), f"must return a static one!"

        return static

    def __matmul__(self, other):
        """self @ other"""
        if other.__class__ is MseHttDynamicLocalMatrix:

            mat_list = [self, other]

            return MseHttDynamicLocalMatrix(mat_list)

        elif other.__class__ in (MseHttDynamicCochainVector, MseHttDynamicLocalVector):

            _matmul = _MatMulDynamicLocalVector(self, other)

            return MseHttDynamicLocalVector(_matmul)

        else:
            raise NotImplementedError(f"{self.__class__.__name__} cannot @ {other.__class__.__name__}")


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
