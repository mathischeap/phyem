# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen

from tools.functions.space._2d.geometrical_functions.anticlockwise_arc import ArcAntiClockWise
from tools.functions.space._2d.geometrical_functions.clockwise_arc import ArcClockWise
from tools.functions.space._2d.geometrical_functions.straight_line import StraightLine


class GeoFunc2Parser(Frozen):
    """"""
    def __init__(self, geo_name, geo_parameters):
        """"""
        if geo_name == 'straight line':
            self._gf = StraightLine(*geo_parameters)
        elif geo_name == 'clockwise arc':
            self._gf = ArcClockWise(*geo_parameters)
        elif geo_name == 'anticlockwise arc':
            self._gf = ArcAntiClockWise(*geo_parameters)
        else:
            raise NotImplementedError(f'2d geo function {geo_name} is not implemented')
        self._name = geo_name
        self._freeze()

    def gamma(self, r):
        """"""
        return self._gf.gamma(r)

    def dgamma(self, r):
        """"""
        return self._gf.dgamma(r)
