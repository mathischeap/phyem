# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:23 PM on 5/25/2023
"""
from tools.frozen import Frozen


class MsePyBoundarySectionElements(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._initialize_elements()
        self._freeze()

    def _initialize_elements(self):
        """"""
        base = self._mesh.base
        region_map = self._mesh.manifold.regions.map
        base_map = base.manifold.regions.map
        base_elements_numbering = base.elements._numbering

        print(region_map, base_map)


