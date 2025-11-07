# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen

from msehtt.static.mesh.great.config.specific_meshes.periodic_slice import periodic_slice


class Predefined_Specific_Meshes(Frozen):
    """"""
    def __init__(self):
        """"""
        self._freeze()

    @classmethod
    def defined(cls):
        r""""""
        return {
            'periodic slice': periodic_slice,
        }
