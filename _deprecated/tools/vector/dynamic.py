# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehy.tools.vector.static.local import IrregularStaticLocalVector


class IrregularDynamicLocalVector(Frozen):
    """"""

    def __init__(self, vec_caller):
        """"""
        if callable(vec_caller):
            self._vec_caller = vec_caller
        else:
            raise NotImplementedError()

        self._freeze()

    def __call__(self, *args, **kwargs):
        static = self._vec_caller(*args, **kwargs)
        # it should be a static local vector or its subclass
        assert isinstance(static, IrregularStaticLocalVector) or issubclass(static, IrregularStaticLocalVector)
        return static


class IrregularDynamicCochainVector(IrregularDynamicLocalVector):
    """"""

    def __init__(self, rf, dynamic_cochain):
        """"""
        self._f = rf
        super().__init__(dynamic_cochain)
        self._freeze()
