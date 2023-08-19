# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MsePyMeshVisualizeVTK(Frozen):
    """"""

    def __init__(self, mesh):
        self._mesh = mesh
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
