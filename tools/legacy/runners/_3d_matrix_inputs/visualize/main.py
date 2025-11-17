# -*- coding: utf-8 -*-

from phyem.tools.frozen import Frozen
from phyem.tools.legacy.runners._3d_matrix_inputs.visualize.quick import ___SPM3IRV_quick___


# noinspection PyUnusedLocal
class ___SlaveParallelMatrix3dInputRunnerVisualize___(Frozen):
    """We have this just to make that we can call visualize without indicate RANK."""
    def __init__(self, pm3ir):
        self._pm3ir_ = pm3ir
        self._quick_ = ___SPM3IRV_quick___()
        self._freeze()

    def __call__(self, *args, **kwargs):
        return None

    @property
    def quick(self):
        """ Access to the quick visualization methods."""
        return self._quick_

    @staticmethod
    def plot(*args, **kwargs):
        return None

    @staticmethod
    def semilogx(*args, **kwargs):
        return None

    @staticmethod
    def semilogy(*args, **kwargs):
        return None

    @staticmethod
    def loglog(*args, **kwargs):
        return None
