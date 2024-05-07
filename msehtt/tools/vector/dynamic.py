# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.tools.vector.static.local import MseHttStaticLocalVector


class MseHttDynamicLocalVector(Frozen):
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
        assert isinstance(static, MseHttStaticLocalVector) or issubclass(static, MseHttStaticLocalVector)
        return static
