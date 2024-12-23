# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.manifold.predefined.chaotic import chaotic
from msehtt.static.manifold.predefined.rectangle import rectangle


class Predefined_Msehtt_Manifold_Distributor(Frozen):
    """"""
    def __init__(self):
        """"""
        self._freeze()

    @classmethod
    def defined_manifolds(cls):
        r""""""
        return {
            'chaotic': chaotic,
            'rectangle': rectangle,
        }
