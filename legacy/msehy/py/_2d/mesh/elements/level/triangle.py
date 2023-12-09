# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen
from tools.functions.space._2d.distance import distance as _2_distance
from tools.functions.space._2d.angle import angle as _2_angle

_global_topology = {}
_topology_signature_pool = {}
_metric_signature_pool = {}


class MseHyPy2MeshElementsLevelTriangle(Frozen):
    """"""

    def __init__(self, level_triangles, index):
        self._triangles = level_triangles
        self._level = level_triangles._level
        self._index = index
        # --- for a level 0 triangle, its index is like: ---
        # 'e=i' where e is the base element number and i in {0, 1, 2, 3}
        # `i` represents:
        #      y
        #    ^
        #    |________
        #    |\  1  /|
        #    | \   / |
        #    |  \ /  |
        #    |2 / \ 0|
        #    | / 3 \ |
        #    |/_____\|_____> x
        #
        # ------ for deeper triangle, index is like: ---
        # 'e=i-j-k-...', i, j, k, ... in {0, 1}. Since each triangle is split into two on the next level,
        # and we indicate these two by 0, 1 and using the right hand rule (anti-clock-wise).
        self._ct = None
        self.___pair_to___ = False
        self._characters = None
        self._topology_signature = None
        self.___topology___ = None
        self._metric_signature = None
        self._freeze()

    def __repr__(self):
        """"""
        return (f"<hy-Triangle {self._index} on G[{self._level.generation}] "
                f"level[{self._level._level_num}] UPON {self._level.background}>")

    @property
    def level_num(self):
        """this triangle is on this level."""
        return self._level._level_num

    @property
    def local_map(self):
        return self._triangles.local_map[self._index]

    @property
    def pair_to(self):
        """The object (another same level triangle, a triangle of the last level, a base element, or boundary)
        at the bottom of a triangle.

        tuple: If the bottom is a base element, we return a tuple (base element number, m, n) where m, n indicate
        the bottom is which face of the base element.

        str: If the bottom is another triangle (on the same or not level) , just return the index of this
        triangle which shares an edge with self-triangle.

        None: If the bottom is on mesh boundary, then return None.

        Only when pair_to return str or None, this triangle can be future divided into two smaller ones.

        """
        if self.___pair_to___ is False:
            index = self._index
            if self.level_num == 0:
                base_element, sequence = index.split('=')
                # sequence means:
                #     y
                #    ^
                #    |________
                #    |\  1  /|
                #    | \   / |
                #    |  \ /  |
                #    |2 / \ 0|
                #    | / 3 \ |
                #    |/_____\|_____> x
                #
                m, n = None, None
                pair_sequence = -1
                match sequence:
                    case '0':
                        m, n = 0, 1   # x + side triangle
                        pair_sequence = 2
                    case '1':
                        m, n = 1, 1   # y + side triangle
                        pair_sequence = 3
                    case '2':
                        m, n = 0, 0   # x - side triangle
                        pair_sequence = 0
                    case '3':
                        m, n = 1, 0   # y - side triangle
                        pair_sequence = 1

                base_element_map = self._level.background.elements.map[int(base_element)]
                bottom_index = m * 2 + n
                bottom_base_element_object = base_element_map[bottom_index]

                if bottom_base_element_object == -1:
                    self.___pair_to___ = None
                else:
                    if bottom_base_element_object in self._level._refining_elements:

                        self.___pair_to___ = rf"{bottom_base_element_object}={pair_sequence}"

                    else:
                        if n == 0:
                            n = 1
                        else:
                            n = 0
                        self.___pair_to___ = (bottom_base_element_object, m, n)

            else:
                obj = self._triangles.local_map[self._index][0]
                if obj is None:
                    return None
                else:
                    return obj[0]

        return self.___pair_to___

    @property
    def ct(self):
        """coordinate transformation of this triangle."""
        if self._ct is None:
            self._ct = _TriangleCoordinateTransformation(self)
        return self._ct

    @property
    def characters(self):
        """

        characters = [(xt, yt), (xb, yb)] (relative to reference coordinate system) for example,

        ^ y
        |
        |
           . t
          /|\
         / |h\
        /__|__\
           b
        --------------> x

        return base element number, [(xt, yt), (xb, yb)].
        """
        if self._characters is None:
            self._characters = self._triangles._find_characters_of_triangle(self._index)
            assert self._characters[0] == int(self._index.split('=')[0]), f"safety check!"
        return self._characters

    @property
    def _base_element(self):
        """the base msepy element object"""
        return self._level.background.elements[self.characters[0]]

    @property
    def region(self):
        """this triangle is in this region."""
        return self._base_element.region

    @property
    def top(self):
        """coordinates of top vertex."""
        return self.characters[1][0]

    @property
    def bottom(self):
        """coordinates of bottom vertex (center of the bottom (longest, facing the perp corner) edge)."""
        return self.characters[1][1]

    @property
    def topology_signature(self):
        if self._topology_signature is None:
            x0, y0 = self.top
            x1, y1 = self.bottom
            x0 = round(x0, 3)
            x1 = round(x1, 3)
            y0 = round(y0, 3)
            y1 = round(y1, 3)
            signature = f"={x0}-{x1}+{y0}-{y1}"
            if signature in _topology_signature_pool:
                pass
            else:
                _topology_signature_pool[signature] = signature

            self._topology_signature = _topology_signature_pool[signature]

        return self._topology_signature

    @property
    def _topology(self):
        if self.___topology___ is None:
            if self.topology_signature in _global_topology:
                pass
            else:
                topology = _TriangleTopology(self.top, self.bottom)
                _global_topology[self.topology_signature] = topology
            self.___topology___ = _global_topology[self.topology_signature]
        return self.___topology___

    @property
    def metric_signature(self):
        """"""
        if self._metric_signature is None:
            base_metric_signature = self._base_element.metric_signature
            if base_metric_signature is None:
                self._metric_signature = id(self)
            else:
                signature = '>' + base_metric_signature + self.topology_signature
                if signature in _metric_signature_pool:
                    pass
                else:
                    _metric_signature_pool[signature] = signature
                self._metric_signature = _metric_signature_pool[signature]
        return self._metric_signature

    @property
    def angle(self):
        """the angle the center line of the perp corner is facing (towards the bottom center)."""
        return self._topology.angle

    @property
    def angle_degree(self):
        return self._topology.angle_degree

    @property
    def height(self):
        return self._topology.height

    @property
    def length_side(self):
        return self._topology.length_side

    @property
    def length_bottom(self):
        return self._topology.length_bottom

    @property
    def vertex0(self):
        """coordinates of vertex #0: the vertex on the line of angle `self.angle - pi/4`."""
        return self._topology.vertex0

    @property
    def vertex1(self):
        """coordinates of vertex #0: the vertex on the line of angle `self.angle + pi/4`."""
        return self._topology.vertex1


class _TriangleCoordinateTransformation(Frozen):
    """"""
    def __init__(self, triangle):
        """"""
        self._triangle = triangle
        self._topo_ct = triangle._topology._ct
        self._base_ct = triangle._base_element.ct
        self._edges = {}
        self._freeze()

    def __repr__(self):
        return f"<CT of {self._triangle}>"

    def edge(self, edge_index):
        """

        Parameters
        ----------
        edge_index : {'b', 0, 1}

        Returns
        -------

        """
        if edge_index in self._edges:
            pass
        else:
            self._edges[edge_index] = _TriangleEdgeCT(
                self, edge_index
            )
        return self._edges[edge_index]

    def mapping(self, xi, et):
        """"""
        r, s = self._topo_ct.mapping(xi, et)
        return self._base_ct.mapping(r, s)

    def Jacobian_matrix(self, xi, et):
        """"""
        dr, ds = self._topo_ct.Jacobian_matrix(xi, et)
        dr_dxi, dr_det = dr
        ds_dxi, ds_det = ds

        r, s = self._topo_ct.mapping(xi, et)
        dx, dy = self._base_ct.Jacobian_matrix(r, s)
        dx_dr, dx_ds = dx
        dy_dr, dy_ds = dy

        dx_dxi = dx_dr * dr_dxi + dx_ds * ds_dxi
        dx_det = dx_dr * dr_det + dx_ds * ds_det

        dy_dxi = dy_dr * dr_dxi + dy_ds * ds_dxi
        dy_det = dy_dr * dr_det + dy_ds * ds_det

        return ([dx_dxi, dx_det],
                [dy_dxi, dy_det])

    def Jacobian(self, xi, et):
        """"""
        J = self.Jacobian_matrix(xi, et)
        return J[0][0]*J[1][1] - J[0][1]*J[1][0]

    def metric(self, xi, et):
        """
        The metric ``g:= det(G):=(det(J))**2``. Since our Jacobian and inverse of Jacobian are both square,
        we know that the metric ``g`` is equal to square of ``det(J)``. ``g = (det(J))**2`` is due to the
        fact that the Jacobian matrix is square. The definition of ``g`` usually is given
        as ``g:= det(G)`` where ``G`` is the metric matrix, or metric tensor.

        """
        detJ = self.Jacobian(xi, et)
        return detJ ** 2

    def inverse_Jacobian_matrix(self, xi, et):
        """The inverse Jacobian matrix. """
        J = self.Jacobian_matrix(xi, et)
        Jacobian = J[0][0]*J[1][1] - J[0][1]*J[1][0]
        reciprocalJacobian = 1 / Jacobian
        del Jacobian
        iJ00 = + reciprocalJacobian * J[1][1]
        iJ01 = - reciprocalJacobian * J[0][1]
        iJ10 = - reciprocalJacobian * J[1][0]
        iJ11 = + reciprocalJacobian * J[0][0]
        return ([iJ00, iJ01],
                [iJ10, iJ11])

    def inverse_Jacobian(self, xi, et):
        """the Determinant of the inverse Jacobian matrix."""
        ijm = self.inverse_Jacobian_matrix(xi, et)
        inverse_Jacobian = (ijm[0][0] * ijm[1][1] - ijm[0][1] * ijm[1][0])
        return inverse_Jacobian

    def metric_matrix(self, xi, et):
        """"""
        jm = self.Jacobian_matrix(xi, et)
        m = n = 2
        G = [[None for _ in range(n)] for __ in range(n)]
        for i in range(n):
            for j in range(i, n):
                # noinspection PyTypeChecker
                G[i][j] = jm[0][i] * jm[0][j]
                for L in range(1, m):
                    G[i][j] += jm[L][i] * jm[L][j]
                if i != j:
                    G[j][i] = G[i][j]
        return G

    def inverse_metric_matrix(self, xi, et):
        """"""
        ijm = self.inverse_Jacobian_matrix(xi, et)
        m = n = 2
        iG = [[None for _ in range(m)] for __ in range(m)]
        for i in range(m):
            for j in range(i, m):
                # noinspection PyTypeChecker
                iG[i][j] = ijm[i][0] * ijm[j][0]
                for L in range(1, n):
                    iG[i][j] += ijm[i][L] * ijm[j][L]
                if i != j:
                    iG[j][i] = iG[i][j]
        return iG


class _TriangleEdgeCT(Frozen):
    """"""
    def __init__(self, ct, edge_index):
        """"""
        assert edge_index in ('b', 0, 1), f"edge index = {edge_index} wrong; it must be among 'b', 0, 1."
        self._ct = ct
        self._edge_index = edge_index
        self._freeze()

    def mapping(self, xi):
        """"""
        ei = self._edge_index
        ones = np.ones_like(xi)
        if ei == 'b':  # bottom edge
            return self._ct.mapping(ones, xi)
        elif ei == 0:  # edge0
            return self._ct.mapping(xi, -ones)
        elif ei == 1:  # edge1
            return self._ct.mapping(xi, ones)
        else:
            raise Exception()

    def Jacobian_matrix(self, xi):
        """"""
        e = self._edge_index
        t = xi
        ones = np.ones_like(t)
        if e == 'b':  # x+
            JM = self._ct.Jacobian_matrix(ones, t)
            return JM[0][1], JM[1][1]

        else:
            if e == 0:  # y-
                JM = self._ct.Jacobian_matrix(t, -ones)
            elif e == 1:  # y+
                JM = self._ct.Jacobian_matrix(t, ones)
            else:
                raise Exception()

            return JM[0][0], JM[1][0]

    def outward_unit_normal_vector(self, xi):
        """The outward unit norm vector (vec{n})."""
        JM = self.Jacobian_matrix(xi)

        x, y = JM

        e = self._edge_index

        if e == 1:
            vx, vy = -y, x
        else:
            vx, vy = y, -x

        magnitude = np.sqrt(vx**2 + vy**2)

        return vx / magnitude, vy / magnitude


class _TriangleTopology(Frozen):
    """"""
    def __init__(self, coo_top, coo_bottom):
        """"""
        self._top = coo_top
        self._bottom = coo_bottom
        self._angle = None
        self._angle_degree = None
        self._height = None
        self.___height_sqrt2__ = None
        self._length_bottom = None
        self._vertex0 = None
        self._vertex1 = None
        self._ct = _TopologyCoordinateTransformation(self)
        self._freeze()

    @property
    def ct(self):
        """"""
        return self._ct

    @property
    def top(self):
        """coordinates of top vertex."""
        return self._top

    @property
    def bottom(self):
        """coordinates of bottom vertex (center of the bottom (longest, facing the perp corner) edge)."""
        return self._bottom

    @property
    def angle(self):
        """the angle the center line of the perp corner is facing (towards the bottom center)."""
        if self._angle is None:
            angle = _2_angle(self._top, self._bottom)
            if np.isclose(angle, np.pi*2):
                angle = 0
            else:
                pass
            self._angle = angle
        return self._angle

    @property
    def angle_degree(self):
        if self._angle_degree is None:
            self._angle_degree = int(self.angle * 180 / np.pi)
        return self._angle_degree

    @property
    def height(self):
        if self._height is None:
            self._height = _2_distance(self._top, self._bottom)
        return self._height

    @property
    def length_side(self):
        if self.___height_sqrt2__ is None:
            self.___height_sqrt2__ = self.height * np.sqrt(2)
        return self.___height_sqrt2__

    @property
    def length_bottom(self):
        if self._length_bottom is None:
            self._length_bottom = 2 * self.height
        return self._length_bottom

    @property
    def vertex0(self):
        """coordinates of vertex #0: the vertex on the line of angle `self.angle - pi/4`."""
        if self._vertex0 is None:
            x, y = self._top
            angle = self.angle - np.pi/4
            self._vertex0 = (
                x + self.length_side * np.cos(angle),
                y + self.length_side * np.sin(angle)
            )
        return self._vertex0

    @property
    def vertex1(self):
        """coordinates of vertex #1: the vertex on the line of angle `self.angle + pi/4`."""
        if self._vertex1 is None:
            x, y = self._top
            angle = self.angle + np.pi/4
            self._vertex1 = (
                x + self.length_side * np.cos(angle),
                y + self.length_side * np.sin(angle)
            )
        return self._vertex1


from tools.functions.space._2d.transfinite import TransfiniteMapping
from tools.miscellaneous.ndarray_cache import ndarray_key_comparer, add_to_ndarray_cache


class _TopologyCoordinateTransformation(Frozen):
    """ 'x-' side is mapped into the top vertex. 'x+' side is mapped into the bottom edge.
    'y-' side is mapped into edge0, 'y+' side is mapped into edge 1. 'edge0' -> 'edge1' is
    the right-hand-rule rotational direction.

    """
    def __init__(self, topo):
        """
        """
        self._tp = topo
        self._xt, self._yt = topo.top
        self._x0, self._y0 = topo.vertex0
        self._x1, self._y1 = topo.vertex1

        self._x01 = self._x1 - self._x0
        self._y01 = self._y1 - self._y0
        self._xt0 = self._x0 - self._xt
        self._yt0 = self._y0 - self._yt
        self._xt1 = self._x1 - self._xt
        self._yt1 = self._y1 - self._yt

        self._tfm = TransfiniteMapping(
            [self._mapping_edge0, self._mapping_bottom, self._mapping_edge1, self._mapping_top],
            [self._Jacobian_edge0, self._Jacobian_bottom, self._Jacobian_edge1, self._Jacobian_top]
        )
        self._cache_0_tp_mp = dict()
        self._cache_1_tp_jm = dict()

        self._freeze()

    def _mapping_top(self, o):
        """'x-' side into the top vertex."""
        return (self._xt * np.ones_like(o),
                self._yt * np.ones_like(o))

    def _mapping_bottom(self, o):
        """'x+' side into the bottom edge."""
        return (self._x0 + o * self._x01,
                self._y0 + o * self._y01)

    def _mapping_edge0(self, o):
        """'y-' side into edge 0."""
        return (self._xt + o * self._xt0,
                self._yt + o * self._yt0)

    def _mapping_edge1(self, o):
        """'y-' side into edge 0."""
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
        """xi, et be in [-1, 1]."""
        cached, cache_data = ndarray_key_comparer(self._cache_0_tp_mp, [xi, et])
        if cached:
            return cache_data
        else:
            pass
        xi_ = (xi + 1) / 2
        et_ = (et + 1) / 2
        mp = self._tfm.mapping(xi_, et_)
        add_to_ndarray_cache(self._cache_0_tp_mp, [xi, et], mp, maximum=5)
        return mp

    def Jacobian_matrix(self, xi, et):
        """xi, s be in [-1, 1]."""
        cached, cache_data = ndarray_key_comparer(self._cache_1_tp_jm, [xi, et])
        if cached:
            return cache_data
        else:
            pass
        xi_ = (xi + 1) / 2
        et_ = (et + 1) / 2
        jm = ((0.5 * self._tfm.dx_dr(xi_, et_), 0.5 * self._tfm.dx_ds(xi_, et_)),
              (0.5 * self._tfm.dy_dr(xi_, et_), 0.5 * self._tfm.dy_ds(xi_, et_)))
        add_to_ndarray_cache(self._cache_1_tp_jm, [xi, et], jm, maximum=5)
        return jm
