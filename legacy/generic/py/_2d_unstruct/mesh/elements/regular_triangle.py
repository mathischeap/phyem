# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from legacy.generic.py._2d_unstruct.mesh.elements.base import Element, CoordinateTransformation
from tools.quadrature import Quadrature


class RegularTriangle(Element):
    r"""
    The local indices of regular quadrilateral vertices and edges.

    r---------------> et

         0
        / \
       /   \
      /e0   \e2
     /       \
    /_________\
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
            coordinates of vertex 0, 1, 2:
                (x0, y0) = element_coordinates[0]
                (x1, y1) = element_coordinates[1]
                (x2, y2) = element_coordinates[2]

        """
        self._coo = element_coordinates
        self._ct = None
        self._area = None
        x0, y0 = element_coordinates[0]
        x1, y1 = element_coordinates[1]
        x2, y2 = element_coordinates[2]
        vec1 = np.array([x1-x0, y1-y0])
        vec2 = np.array([x2-x0, y2-y0])
        vec1 = list(np.round(vec1, 6))
        vec2 = list(np.round(vec2, 6))
        self._metric_signature = f"{vec1}{vec2}"
        self._edges_ct = {
            0: Triangle_Edge_CT(self, 0),
            1: Triangle_Edge_CT(self, 1),
            2: Triangle_Edge_CT(self, 2),
        }
        self._freeze()

    @property
    def type(self):
        return 't'

    @property
    def metric_signature(self):
        return self._metric_signature

    @property
    def orthogonal(self):
        return True

    @property
    def ct(self):
        if self._ct is None:
            self._ct = _RegularTriangleCoordinateTransformation(
                self._coo, self.metric_signature
            )
        return self._ct

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

    @classmethod
    def inner_orientations(cls, j):
        r"""
        ---------------> et

             0
            / \
           /   \
          /V   V\
         /       \
        /_________\
        1    >    2


        |
        |
        |
        V xi
        """
        return '++-'[j]

    @classmethod
    def outer_orientations(cls, j):
        r"""
        ---------------> et

             0
            / \
           /   \
          />    \>
         /       \
        /_________\
        1    v    2


        |
        |
        |
        V xi
        """
        return '+--'[j]

    def _plot_lines(self, density):
        """"""
        space = np.linspace(-1, 1, 2)
        ones = np.ones_like(space)
        y0_edge = (space, -ones)
        x1_edge = (ones, space)
        y1_edge = (space, ones)
        coo_edge0 = self.ct.mapping(*y0_edge)
        coo_edge1 = self.ct.mapping(*x1_edge)
        coo_edge2 = self.ct.mapping(*y1_edge)

        # x = np.array([-1, 0])
        # y = np.array([0, 0])
        # singular_line = self.ct.mapping(x, y)

        return coo_edge0, coo_edge1, coo_edge2  # singular_line

    def edge_ct(self, edge_index):
        return self._edges_ct[edge_index]


from tools.functions.space._2d.transfinite import TransfiniteMapping


class _RegularTriangleCoordinateTransformation(CoordinateTransformation):
    """"""
    def __init__(self, coo, metric_signature):
        """"""
        super().__init__(metric_signature)
        self._xt, self._yt = coo[0]
        self._x0, self._y0 = coo[1]
        self._x1, self._y1 = coo[2]

        self._x01 = self._x1 - self._x0
        self._y01 = self._y1 - self._y0
        self._xt0 = self._x0 - self._xt
        self._yt0 = self._y0 - self._yt
        self._xt1 = self._x1 - self._xt
        self._yt1 = self._y1 - self._yt

        self._tf = TransfiniteMapping(
            [self._mapping_edge0, self._mapping_bottom, self._mapping_edge1, self._mapping_top],
            [self._Jacobian_edge0, self._Jacobian_bottom, self._Jacobian_edge1, self._Jacobian_top]
        )

    def _mapping_top(self, o):
        """'x-' side into the top vertex. """
        return (self._xt * np.ones_like(o),
                self._yt * np.ones_like(o))

    def _mapping_bottom(self, o):
        """'x+' side into the bottom edge."""
        return (self._x0 + o * self._x01,
                self._y0 + o * self._y01)

    def _mapping_edge0(self, o):
        """'y-' side into edge 0"""
        return (self._xt + o * self._xt0,
                self._yt + o * self._yt0)

    def _mapping_edge1(self, o):
        """'y+' side into edge 1"""
        return (self._xt + o * self._xt1,
                self._yt + o * self._yt1)

    @staticmethod
    def _Jacobian_top(o):
        """"""
        z0 = np.zeros_like(o)
        return z0, z0

    def _Jacobian_bottom(self, o):
        """"""
        return (self._x01 * np.ones_like(o),
                self._y01 * np.ones_like(o))

    def _Jacobian_edge0(self, o):
        """"""
        return (self._xt0 * np.ones_like(o),
                self._yt0 * np.ones_like(o))

    def _Jacobian_edge1(self, o):
        """"""
        return (self._xt1 * np.ones_like(o),
                self._yt1 * np.ones_like(o))

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
_t_r_cache = {}


class Triangle_Edge_CT(Frozen):
    def __init__(self, triangle, edge_index):
        self._triangle = triangle
        self._edge_index = edge_index
        self._freeze()

    def _parse_r(self, r):
        """"""
        ei = self._edge_index
        cached, xi_et = ndarray_key_comparer(_t_r_cache, [r], check_str=str(ei))
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
            else:
                raise Exception()
            add_to_ndarray_cache(_t_r_cache, [r], xi_et, check_str=str(ei), maximum=8)

        return xi_et

    def mapping(self, r):
        """"""
        xi, et = self._parse_r(r)
        return self._triangle.ct.mapping(xi, et)

    def Jacobian_matrix(self, r):
        """"""
        xi, et = self._parse_r(r)
        JM = self._triangle.ct.Jacobian_matrix(xi, et)
        ei = self._edge_index

        if ei == 1:  # x-axis-direction
            return JM[0][1], JM[1][1]

        elif ei in (0, 2):  # y-axis-direction
            return JM[0][0], JM[1][0]

        else:
            raise Exception()

    def outward_unit_normal_vector(self, r):
        """"""
        JM = self.Jacobian_matrix(r)

        x, y = JM

        ei = self._edge_index

        if ei == 2:
            vx, vy = -y, x
        else:
            vx, vy = y, -x

        magnitude = np.sqrt(vx**2 + vy**2)

        return vx / magnitude, vy / magnitude
