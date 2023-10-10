# -*- coding: utf-8 -*-
r"""
"""
from generic.py._2d_unstruct.mesh.elements.base import Element, CoordinateTransformation
from tools.quadrature import Quadrature
import numpy as np


class RegularQuadrilateral(Element):
    """
    The local indices of regular quadrilateral vertices and edges.

    ---------------> et

    0____e3___3
    |         |
    | e0      |e2
    |         |
    |_________|
    1    e1   2

    |
    |
    |
    V xi

    """

    def __init__(self, element_coordinates):
        """

        Parameters
        ----------
        element_coordinates :
            coordinates of vertex 0, 1, 2, 3:
                (x0, y0) = element_coordinates[0]
                (x1, y1) = element_coordinates[1]
                (x2, y2) = element_coordinates[2]
                (x3, y3) = element_coordinates[3]

        """
        self._coo = element_coordinates
        self._ct = None
        self._area = None
        x0, y0 = element_coordinates[0]
        x1, y1 = element_coordinates[1]
        x2, y2 = element_coordinates[2]
        x3, y3 = element_coordinates[3]
        vec1 = np.array([x1-x0, y1-y0])
        vec2 = np.array([x2-x0, y2-y0])
        vec3 = np.array([x3-x0, y3-y0])
        vec1 = tuple(np.round(vec1, 6))
        vec2 = tuple(np.round(vec2, 6))
        vec3 = tuple(np.round(vec3, 6))
        self._metric_signature = f"{vec1}{vec2}{vec3}"
        self._edges_ct = {
            0: Quadrilateral_Edge_CT(self, 0),
            1: Quadrilateral_Edge_CT(self, 1),
            2: Quadrilateral_Edge_CT(self, 2),
            3: Quadrilateral_Edge_CT(self, 3),
        }
        self._freeze()

    @property
    def type(self):
        return 'q'

    @property
    def metric_signature(self):
        return self._metric_signature

    @property
    def orthogonal(self):
        return True

    @property
    def ct(self):
        if self._ct is None:
            self._ct = _RegularQuadrilateralCoordinateTransformation(
                self._coo, self.metric_signature
            )
        return self._ct

    @classmethod
    def inner_orientations(cls, j):
        """
        ---------------> et

        0____>____3
        |         |
        |v        |v
        |         |
        |_________|
        1    >    2

        |
        |
        |
        V xi
        """
        return '++--'[j]

    @classmethod
    def outer_orientations(cls, j):
        """
        ---------------> et

        0____v____3
        |         |
        |>        |>
        |         |
        |_________|
        1    v    2

        |
        |
        |
        V xi
        """
        return '+--+'[j]

    @property
    def area(self):
        """the area of this element."""
        if self._area is None:
            quad = Quadrature([5, 5], category='Gauss')
            nodes = quad.quad_ndim[:-1]
            weights = quad.quad_ndim[-1]
            detJ = self.ct.Jacobian(*nodes)
            self._area = np.sum(detJ * weights)
        return self._area

    def _plot_lines(self, density):
        """"""
        space = np.linspace(-1, 1, 2)
        ones = np.ones_like(space)
        x0_edge = (-ones, space)
        x1_edge = (ones, space)
        y0_edge = (space, -ones)
        y1_edge = (space, ones)
        edge0 = self.ct.mapping(*y0_edge)
        edge1 = self.ct.mapping(*x1_edge)
        edge2 = self.ct.mapping(*y1_edge)
        edge3 = self.ct.mapping(*x0_edge)
        return edge0, edge1, edge2, edge3

    def edge_ct(self, edge_index):
        return self._edges_ct[edge_index]


from tools.functions.space._2d.transfinite import TransfiniteMapping
from tools.functions.space._2d.geometrical_functions.parser import GeoFunc2Parser


class _RegularQuadrilateralCoordinateTransformation(CoordinateTransformation):
    """"""
    def __init__(self, coo, metric_signature):
        """"""
        super().__init__(metric_signature)
        geo_y0 = ['straight line', [coo[0], coo[1]]]
        geo_x1 = ['straight line', [coo[1], coo[2]]]
        geo_y1 = ['straight line', [coo[3], coo[2]]]
        geo_x0 = ['straight line', [coo[0], coo[3]]]
        geo_y0 = GeoFunc2Parser(*geo_y0)
        geo_x1 = GeoFunc2Parser(*geo_x1)
        geo_y1 = GeoFunc2Parser(*geo_y1)
        geo_x0 = GeoFunc2Parser(*geo_x0)
        gamma = [
            geo_y0.gamma,
            geo_x1.gamma,
            geo_y1.gamma,
            geo_x0.gamma,
        ]
        d_gamma = [
            geo_y0.dgamma,
            geo_x1.dgamma,
            geo_y1.dgamma,
            geo_x0.dgamma,
        ]
        self._tf = TransfiniteMapping(gamma, d_gamma)

    def mapping(self, xi, et):
        """"""
        r = (xi + 1) / 2
        s = (et + 1) / 2
        return self._tf.mapping(r, s)

    def ___Jacobian_matrix___(self, xi, et):
        """ r, s be in [-1, 1]. """
        r = (xi + 1) / 2
        s = (et + 1) / 2
        return ((0.5 * self._tf.dx_dr(r, s), 0.5 * self._tf.dx_ds(r, s)),
                (0.5 * self._tf.dy_dr(r, s), 0.5 * self._tf.dy_ds(r, s)))


from tools.frozen import Frozen
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer
_q_r_cache = {}


class Quadrilateral_Edge_CT(Frozen):
    def __init__(self, quadrilateral, edge_index):
        self._quadrilateral = quadrilateral
        self._edge_index = edge_index
        self._freeze()

    def _parse_r(self, r):
        """"""
        ei = self._edge_index
        cached, xi_et = ndarray_key_comparer(_q_r_cache, [r], check_str=str(ei))
        if cached:
            pass
        else:
            if ei == 0:
                xi_et = [
                    r, -np.ones_like(r)
                ]
            elif ei == 1:
                xi_et = [
                    np.ones_like(r), r
                ]
            elif ei == 2:
                xi_et = [
                    r, np.ones_like(r)
                ]
            elif ei == 3:
                xi_et = [
                    -np.ones_like(r), r
                ]
            else:
                raise Exception()
            add_to_ndarray_cache(_q_r_cache, [r], xi_et, check_str=str(ei), maximum=8)
        return xi_et

    def mapping(self, r):
        """"""
        xi, et = self._parse_r(r)
        return self._quadrilateral.ct.mapping(xi, et)

    def Jacobian_matrix(self, r):
        """"""
        xi, et = self._parse_r(r)
        JM = self._quadrilateral.ct.Jacobian_matrix(xi, et)
        ei = self._edge_index

        if ei in (1, 3):  # x-direction
            return JM[0][1], JM[1][1]

        elif ei in (0, 2):  # x-direction
            return JM[0][0], JM[1][0]

        else:
            raise Exception()

    def outward_unit_normal_vector(self, r):
        """"""
        JM = self.Jacobian_matrix(r)

        x, y = JM

        ei = self._edge_index

        if ei in (2, 3):
            vx, vy = -y, x
        else:
            vx, vy = y, -x

        magnitude = np.sqrt(vx**2 + vy**2)

        return vx / magnitude, vy / magnitude
