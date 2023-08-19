# -*- coding: utf-8 -*-
r"""
"""
import sys
if './' not in sys.path:
    sys.path.append('./')

from tools.gen_piece_wise import genpiecewise
from tools.functions.time_space._2d.wrappers.vector import T2dVector
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from tools.frozen import Frozen
from msepy.mesh.main import MsePyMesh
from msepy.manifold.predefined.cylinder_channel import _CylinderChannel


# noinspection PyUnusedLocal
class ConditionsFlowAroundCylinder2(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        if mesh.__class__ is MsePyMesh:
            self._mesh = mesh
            configuration = mesh.manifold._configuration
            assert configuration.__class__ is _CylinderChannel
            r = configuration._r
            dl = configuration._dl
            left_boundary_limit = - 0.9 * dl
            self._left_boundary_limit = left_boundary_limit
        else:
            raise Exception()
        self._freeze()

    @property
    def velocity_initial_condition(self):
        """"""
        return T2dVector(self._0, self._0)

    @property
    def vorticity_initial_condition(self):
        return T2dScalar(self._0)

    @staticmethod
    def _u_in(t, x, y):
        """"""
        return 1 + 0 * x

    @staticmethod
    def _u_other(t, x, y):
        return 0 + 0 * x

    def _u(self, t, x, y):
        return genpiecewise(
            [t, x, y],
            [x < self._left_boundary_limit, x >= self._left_boundary_limit],
            [self._u_in, self._u_other]
        )

    @staticmethod
    def _0(t, x, y):
        return 0 + 0 * x

    @property
    def velocity_boundary_condition(self):
        """"""
        return T2dVector(self._u, self._0)
