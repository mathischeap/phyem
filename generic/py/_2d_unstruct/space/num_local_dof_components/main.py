# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from generic.py._2d_unstruct.space.num_local_dof_components.Lambda import NumLocalDofComponentsLambda


class NumLocalDofComponents(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._bundle = None
        self._freeze()

    def __call__(self, degree):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = NumLocalDofComponentsLambda(self._space)
        return self._Lambda
