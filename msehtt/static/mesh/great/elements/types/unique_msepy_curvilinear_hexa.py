# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement


class MseHtt_GreatMesh_UniqueMsePy_Hexa_Element(MseHttGreatMeshBaseElement):
    """
    This is the curvilinear version of the 3d hexa element.

    Local node numbering:

    back-face: z- face

    _________________________________> y
    |  0                      2
    |  -----------------------
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  1                     3
    x

    forward-face: z+ face

    _________________________________> y
    |  4                      6
    |  -----------------------
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  5                      7
    x

    """
