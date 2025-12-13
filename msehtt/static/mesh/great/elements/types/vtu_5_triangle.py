# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.tools.functions.space._2d.angle import angle
from phyem.tools.functions.space._2d.angles_of_triangle import angles_of_triangle
from phyem.tools.functions.space._2d.distance import distance
from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement


class Vtu5Triangle(MseHttGreatMeshBaseElement):
    r"""
    reference triangle: north edge into a point.

        face west: edge 0
        face east: edge 2
        face south: edge 1
        node 0 (north edge node): (-1, 0)
        node 1 (west-south node): (1, -1)
        node 2 (east-south node): (1, 1)

    ._____________________> et
    |           0        the north edge is collapsed into node 0
    |          /\
    |         /  \                 >   face 0: positive direction: 0->1
    | face 0 /    \ face 2         >>  face 1: positive direction: 1->2
    |       /      \               >>> face 2: positive direction: 0->2
    |      /        \
    |     /__________\
    v     1   face 1  2
    xi

    """

    def __init__(self, element_index, parameters, _map):
        """"""
        a, b, _ = angles_of_triangle(*parameters)
        c = angle(parameters[0], parameters[1])
        d = distance(parameters[0], parameters[1])
        self._metric_signature = (f"5:a%.3f" % round(a, 3) +
                                  "b%.3f" % round(b, 3) +
                                  "c%.3f" % round(c, 3) +
                                  "d%.6f" % round(d, 6))

        super().__init__()
        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._ct = Vtu5Triangle_CT(self)

    @classmethod
    def m(cls):
        r"""the dimensions of the space"""
        return 2

    @classmethod
    def n(cls):
        r"""the dimensions of the element"""
        return 2

    @classmethod
    def _etype(cls):
        r"""vtu cell type 5: triangle"""
        return 5

    @classmethod
    def _find_element_center_coo(cls, parameters):
        r""""""
        return np.sum(np.array(parameters), axis=0) / 3

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<VTU-5 element indexed:{self._index}" + super_repr

    @property
    def metric_signature(self):
        r""""""
        return self._metric_signature

    def _generate_outline_data(self, ddf=None, internal_grid=0):
        r""""""
        x0, y0 = self._parameters[0]
        x1, y1 = self._parameters[1]
        x2, y2 = self._parameters[2]
        line_data_dict = {
            'mn': (self.m(), self.n()),
            'center': self.ct.mapping(0, 0),
            0: ([x0, x1], [y0, y1]),   # face #0; topologically west
            1: ([x1, x2], [y1, y2]),   # face #1; topologically south
            2: ([x2, x0], [y2, y0]),   # face #2; topologically east
        }
        if internal_grid == 0:
            return line_data_dict
        else:
            raise NotImplementedError()

    def _find_element_quality(self):
        r"""Return a factor indicating the quality of the elements.

        When the factor is 0: the element is worst.
        When the factor is 1: the element is best.

        Returns
        -------
        quality: float
            In [0, 1]. 0: Worst; 1: Best.

        """
        parameters = self.parameters
        a, b, c = angles_of_triangle(*parameters)
        quality = np.array([a - 90, b - 90, c - 90])
        quality = np.sqrt(np.sum(quality.dot(quality)))
        quality /= 180
        quality = 1 - quality
        if quality < 1e-8:
            quality = 0
        elif quality > 9.9999:
            quality = 1
        else:
            pass
        return quality

    @classmethod
    def face_setting(cls):
        r"""To show the nodes of faces and the positive direction."""
        return {
            0: (0, 1),   # face #0 is from node 0 -> node 1  (positive direction)
            1: (1, 2),   # face #1 is from node 1 -> node 2  (positive direction)
            2: (0, 2),   # face #2 is from node 0 -> node 2  (positive direction)
        }

    @property
    def faces(self):
        r"""The faces of this element."""
        if self._faces is None:
            self._faces = _VTU5_Triangle_Faces_(self)
        return self._faces

    def ___face_representative_str___(self):
        r""""""
        x = np.array([0, 1, 0])
        y = np.array([-1, 0, 1])
        x, y = self.ct.mapping(x, y)
        return {
            0: r"%.7f-%.7f" % (round(x[0], 7), round(y[0], 7)),
            1: r"%.7f-%.7f" % (round(x[1], 7), round(y[1], 7)),
            2: r"%.7f-%.7f" % (round(x[2], 7), round(y[2], 7)),
        }

    @property
    def edges(self):
        r""""""
        raise Exception(f"vtu triangle element has no edges.")

    def ___edge_representative_str___(self):
        r""""""
        raise Exception(f"vtu triangle element has no edges.")


___invA___ = np.array([
    [-0.5,    0,  0.5],
    [0.25, -0.5, 0.25],
    [0.25,  0.5, 0.25]
])


# ================== CT ==============================================================================


from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElementCooTrans


class Vtu5Triangle_CT(MseHttGreatMeshBaseElementCooTrans):
    r""""""

    def __init__(self, vtu5e):
        r""""""
        super().__init__(vtu5e, vtu5e.metric_signature)

        self._melt()

        # reference triangle: north edge into a point.
        # edge west: edge 0
        # edge east: edge 2
        # edge south: edge 1
        # node 0 (north edge node): (-1, 0)
        # node 1 (west-south node): (1, -1)
        # node 2 (east-south node): (1, 1)
        # A matrix =
        #   [ -1   1   1]
        #   [  0  -1   1]
        #   [  1   1   1]
        # inv(A) =
        #   [ -0.5,    0, 0.5 ]
        #   [ 0.25, -0.5, 0.25]
        #   [ 0.25,  0.5, 0.25]
        #

        X = np.vstack((np.array(
            vtu5e.parameters
        ).T, [1., 1., 1.]))

        T = X @ ___invA___

        self._t00 = T[0, 0]
        self._t01 = T[0, 1]
        self._t02 = T[0, 2]

        self._t10 = T[1, 0]
        self._t11 = T[1, 1]
        self._t12 = T[1, 2]

        self._freeze()

    def mapping(self, xi, et):
        r""""""
        # [-1, 1]^2 -> a reference triangle.
        r = xi
        s = et * (xi + 1) / 2
        # --- then to the physical triangle. ------------------------
        x = self._t00 * r + self._t01 * s + self._t02 * 1
        y = self._t10 * r + self._t11 * s + self._t12 * 1
        return x, y

    def ___Jacobian_matrix___(self, xi, et):
        r""""""
        dx_dr = self._t00
        dx_ds = self._t01

        dy_dr = self._t10
        dy_ds = self._t11

        dr_dxi = 1
        ds_dxi = et / 2
        ds_det = (xi + 1) / 2

        dx_dxi = dx_dr * dr_dxi + dx_ds * ds_dxi
        dx_det = dx_ds * ds_det

        dy_dxi = dy_dr * dr_dxi + dy_ds * ds_dxi
        dy_det = dy_ds * ds_det

        return (
            [dx_dxi, dx_det],
            [dy_dxi, dy_det]
        )


# =========================== faces ===============================================================


class _VTU5_Triangle_Faces_(Frozen):
    r""""""
    def __init__(self, element):
        r""""""
        self._element = element
        self._faces = {}
        self._freeze()

    def __getitem__(self, face_id):
        r"""face_id in (0, 1, 2).

        ______________________> et
        |           0        the north edge is collapsed into node 0
        |          /\
        |         /  \                 >   edge 0: positive direction: 0->1
        | face0  /    \ face2          >>  edge 1: positive direction: 1->2
        |       /      \               >>> edge 2: positive direction: 0->2
        |      /        \
        |     ------------
        v     1   face1   2
        xi

        """
        assert face_id in range(3), f"face id must be in range(3)."
        if face_id not in self._faces:
            self._faces[face_id] = _VTU5_Triangle_OneFace_(self._element, face_id)
        else:
            pass
        return self._faces[face_id]

    def __repr__(self):
        """"""
        return f"<Faces of {self._element}>"


class _VTU5_Triangle_OneFace_(Frozen):
    r""""""
    def __init__(self, element, face_id):
        self._element = element
        self._id = face_id
        self._ct = _VTU5_Triangle_OneFace_CT_(self)
        self._freeze()

    def __repr__(self):
        """"""
        return f"<Face#{self._id} of {self._element}>"

    @property
    def ct(self):
        """Coordinate transformation of this face."""
        return self._ct


from phyem.msehtt.static.mesh.great.elements.types.base import _FaceCoordinateTransformationBase


class _VTU5_Triangle_OneFace_CT_(_FaceCoordinateTransformationBase):
    r""""""

    def __init__(self, face):
        r""""""
        super().__init__(face)
        self._melt()
        fid = face._id
        if fid == 0:
            self._axis, self._start_end = 1, 0
        elif fid == 1:
            self._axis, self._start_end = 0, 1
        elif fid == 2:
            self._axis, self._start_end = 1, 1
        else:
            raise Exception()
        self._freeze()

    def __repr__(self):
        r"""repr"""
        return f"<CT of face #{self._face._id} of {self._element}>"

    def mapping(self, xi):
        r""""""
        m, n = self._axis, self._start_end
        ones = np.ones_like(xi)
        if m == 0:  # x-direction
            if n == 1:  # x+, South, face #1
                return self._element.ct.mapping(ones, xi)
            else:
                raise Exception()
        elif m == 1:  # y-direction
            if n == 0:  # y-, West, face #0
                return self._element.ct.mapping(xi, -ones)
            elif n == 1:  # y+, East, face #2
                return self._element.ct.mapping(xi, ones)
            else:
                raise Exception()
        else:
            raise Exception()

    def Jacobian_matrix(self, xi):
        r""""""
        m, n = self._axis, self._start_end

        ones = np.ones_like(xi)
        if m == 0:  # x-direction
            if n == 1:  # x+, South, face #1
                JM = self._element.ct.Jacobian_matrix(ones, xi)
            else:
                raise Exception()

            return JM[0][1], JM[1][1]

        elif m == 1:  # y-direction
            if n == 0:  # y-, West, face #0
                JM = self._element.ct.Jacobian_matrix(xi, -ones)
            elif n == 1:  # y+, East, face #2
                JM = self._element.ct.Jacobian_matrix(xi, ones)
            else:
                raise Exception()

            return JM[0][0], JM[1][0]

        else:
            raise Exception()

    def outward_unit_normal_vector(self, xi):
        r"""The outward unit norm vector (vec{n})."""
        JM = self.Jacobian_matrix(xi)
        x, y = JM
        m, n = self._axis, self._start_end
        if m == 1 and n == 1:
            vx, vy = -y, x
        else:
            vx, vy = y, -x
        magnitude = np.sqrt(vx**2 + vy**2)
        return vx / magnitude, vy / magnitude

    def is_plane(self):
        r""""""
        return True
