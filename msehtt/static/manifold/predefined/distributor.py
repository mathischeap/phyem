# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.manifold.predefined.chaotic import chaotic
from phyem.msehtt.static.manifold.predefined.rectangle import rectangle
from phyem.msehtt.static.manifold.predefined.quad import quad


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
            'quad': quad,
        }
