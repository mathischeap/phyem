# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen

from tools.functions.space._2d.angle import angle
from tools.functions.space._2d.angles_of_triangle import angles_of_triangle
from tools.functions.space._2d.distance import distance

from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement


class Vtu9Quad(MseHttGreatMeshBaseElement):
    r"""
    _________________________________>  eta
    |  0        face #0       3
    |  -----------------------
    |  |                     |
    |  |                     |
    |  | face #2             |face #3
    |  |                     |
    |  -----------------------
    v  1      face #1        2

    xi

    The labels in _map refers to the for nodes in such a sequence.

    For example, _map = [87, 44, 7561, 156], then it is
    _________________________________> eta
    |  87                    156
    |  -----------------------
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  44                    7561
    xi

    And the real number inherits the numbering.

    """

    def __init__(self, element_index, parameters, _map):
        """"""
        a0, b, c0 = angles_of_triangle(parameters[0], parameters[1], parameters[2])
        a1, c1, _ = angles_of_triangle(parameters[0], parameters[2], parameters[3])
        a = a0 + a1
        c = c0 + c1

        # x0, y0 = parameters[0]
        # x1, y1 = parameters[1]
        # x2, y2 = parameters[2]
        # x3, y3 = parameters[3]
        # print(y2 - y0, x2 - x0)

        d = angle(parameters[0], parameters[2])
        dist = distance(parameters[0], parameters[2])
        self._metric_signature = f"9:a%.3f" % a + "b%.3f" % b + "c%.3f" % c + "d%.3f" % d + "dis%.6f" % dist

        super().__init__()
        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._ct = Vtu9Quad_CT(self)

    @classmethod
    def m(cls):
        """the dimensions of the space"""
        return 2

    @classmethod
    def n(cls):
        """the dimensions of the element"""
        return 2

    @classmethod
    def _etype(cls):
        """vtu cell type 9: quad"""
        return 9

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<VTU-9 element indexed:{self._index}" + super_repr

    @property
    def metric_signature(self):
        """"""
        return self._metric_signature

    def _generate_outline_data(self, ddf=None):
        """"""
        x0, y0 = self._parameters[0]
        x1, y1 = self._parameters[1]
        x2, y2 = self._parameters[2]
        x3, y3 = self._parameters[3]

        return {
            'mn': (self.m(), self.n()),
            'center': self.ct.mapping(0, 0),
            0: ([x0, x3], [y0, y3]),   # face #0
            1: ([x1, x2], [y1, y2]),   # face #1
            2: ([x0, x1], [y0, y1]),   # face #2
            3: ([x3, x2], [y3, y2]),   # face #3
        }

    @classmethod
    def face_setting(cls):
        """To show the nodes of faces and the positive direction."""
        return {
            0: (0, 3),   # face #0 is from node 0 -> node 3  (positive direction)
            1: (1, 2),   # face #1 is from node 1 -> node 2  (positive direction)
            2: (0, 1),   # face #2 is from node 0 -> node 1  (positive direction)
            3: (3, 2),   # face #3 is from node 3 -> node 2  (positive direction)
        }

    @property
    def faces(self):
        """The faces of this element."""
        if self._faces is None:
            self._faces = Quad_Faces(self)
        return self._faces

    def ___face_representative_str___(self):
        r""""""
        x = np.array([-1, 1, 0, 0])
        y = np.array([0, 0, -1, 1])
        x, y = self.ct.mapping(x, y)
        return {
            0: r"%.7f-%.7f" % (x[0], y[0]),
            1: r"%.7f-%.7f" % (x[1], y[1]),
            2: r"%.7f-%.7f" % (x[2], y[2]),
            3: r"%.7f-%.7f" % (x[3], y[3]),
        }

    @property
    def edges(self):
        raise Exception(f"vtu quad element has no edges.")

    def ___edge_representative_str___(self):
        r""""""
        raise Exception(f"vtu quad element has no edges.")

    @classmethod
    def degree_parser(cls, degree):
        """"""
        if isinstance(degree, int):
            p = (degree, degree)
            dtype = 'Lobatto'
        else:
            raise NotImplementedError()
        return p, dtype


___A___ = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 0]
])

___invA___ = np.linalg.inv(___A___)


from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElementCooTrans


class Vtu9Quad_CT(MseHttGreatMeshBaseElementCooTrans):
    r""""""

    def __init__(self, vtu9e):
        """"""
        super().__init__(vtu9e, vtu9e.metric_signature)

        self._melt()

        vec = np.array(vtu9e.parameters)

        x = vec[:, 0]
        y = vec[:, 1]

        alpha = ___invA___ @ x
        beta = ___invA___ @ y

        self._a1, self._a2, self._a3, self._a4 = alpha
        self._b1, self._b2, self._b3, self._b4 = beta

        self._freeze()

    def mapping(self, xi, et):
        """"""
        # to the reference square
        r = (xi + 1) * 0.5
        s = (et + 1) * 0.5
        # to the physical quad
        x = self._a1 + self._a2 * r + self._a3 * s + self._a4 * r * s
        y = self._b1 + self._b2 * r + self._b3 * s + self._b4 * r * s
        return x, y

    def ___Jacobian_matrix___(self, xi, et):
        """"""
        r = (xi + 1) * 0.5
        s = (et + 1) * 0.5

        dx_dr = self._a2 + self._a4 * s
        dx_ds = self._a3 + self._a4 * r

        dy_dr = self._b2 + self._b4 * s
        dy_ds = self._b3 + self._b4 * r

        dr_dxi = 0.5
        ds_det = 0.5

        dx_dxi = dx_dr * dr_dxi
        dx_det = dx_ds * ds_det

        dy_dxi = dy_dr * dr_dxi
        dy_det = dy_ds * ds_det

        return (
            [dx_dxi, dx_det],
            [dy_dxi, dy_det]
        )


# ============ FACES ============================================================================================


class Quad_Faces(Frozen):
    """"""
    def __init__(self, element):
        """"""
        self._element = element
        self._faces = {}
        self._freeze()

    def __getitem__(self, face_id):
        """0, 1, 2, 3.

       _________________________________> eta
        |  0        face #0       3
        |  -----------------------
        |  |                     |
        |  |         (ref)       |
        |  | face #2             |face #3
        |  |                     |
        |  -----------------------
        v  1      face #1        2
        xi

        """
        assert face_id in range(4), f"face id must be in range(4)."
        if face_id not in self._faces:
            self._faces[face_id] = Quad_Face(self._element, face_id)
        else:
            pass
        return self._faces[face_id]

    def __repr__(self):
        """"""
        return f"<Faces of {self._element}>"


class Quad_Face(Frozen):
    """"""
    def __init__(self, element, face_id):
        self._element = element
        self._id = face_id
        self._ct = Quad_Face_CT(self)
        self._freeze()

    def __repr__(self):
        """"""
        return f"<Face#{self._id} of {self._element}>"

    @property
    def ct(self):
        """Coordinate transformation of this face."""
        return self._ct


from msehtt.static.mesh.great.elements.types.orthogonal_rectangle import \
    MseHttGreatMeshOrthogonalRectangleElementFaceCT


class Quad_Face_CT(
    MseHttGreatMeshOrthogonalRectangleElementFaceCT
):
    r""""""
    def __init__(self, face):
        r""""""
        super().__init__(face)
