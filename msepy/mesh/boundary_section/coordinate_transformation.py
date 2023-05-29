# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:03 PM on 5/26/2023
"""
from tools.frozen import Frozen


class MsePyBoundarySectionMeshCooTrans(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._freeze()
