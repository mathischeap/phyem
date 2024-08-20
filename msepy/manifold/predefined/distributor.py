# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from importlib import import_module


class PredefinedMsePyManifoldDistributor(Frozen):
    """"""

    def __init__(self):
        """"""
        self._freeze()

    def __call__(self, object_name):
        """"""
        predefined_path = '.'.join(str(self.__class__).split(' ')[1][1:-2].split('.')[:-2]) + '.' + \
                          object_name

        _module = import_module(predefined_path)

        _the_mf_config = getattr(_module, self._predefined_manifolds()[object_name])

        return _the_mf_config

    @classmethod
    def _predefined_manifolds(cls):
        return {
            'crazy': '_crazy',
            'crazy_multi': '_crazy_multi',
            'backward_step': '_backward_step',
            'cylinder_channel': '_CylinderChannel',
        }
