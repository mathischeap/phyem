# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen

from phyem.msehtt.static.mesh.great.config.specific_meshes.periodic_slice import periodic_slice


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
