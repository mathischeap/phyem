
import sys
if './' not in sys.path:
    sys.path.append('./')

from src.tools.frozen import Frozen
import numpy as np


class MsePyMeshVisualizeVTK(Frozen):
    """"""

    def __init__(self, mesh):
        self._mesh = mesh
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""