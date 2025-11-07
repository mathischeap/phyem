# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
import numpy as np
from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement
from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElementCooTrans


class MseHtt_GreatMesh_Unique_MsePy_Hexahedron_Element(MseHttGreatMeshBaseElement):
    """
    This is the curvilinear version of the 3d hexahedron element.

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


    For example, _map = [87, 44, 156, 7561, 12, 54, 66531, 9985], then the global numbering of mesh nodes of this
    element is

    back-face: z- face

    _________________________________> y
    | 87                     156
    |  -----------------------
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v 44                    7561
    x

    forward-face: z+ face

    _________________________________> y
    | 12                    66531
    |  -----------------------
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v 54                    9985
    x

    """

    def __init__(self, element_index, parameters, _map, msepy_manifold):
        """"""
        self._region = parameters['region']
        self._origin = parameters['origin']
        self._delta = parameters['delta']
        assert msepy_manifold is not None, \
            f"unique msepy curvilinear quadrilateral must have the original msepy manifold."
        self._msepy_manifold = msepy_manifold
        super().__init__()
        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._ct = MseHtt_GreatMesh_Unique_Msepy_Curvilinear_Hexahedron_Element_CooTrans(self, self.metric_signature)

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Unique Msepy Curvilinear hexahedron element indexed:{self._index}" + super_repr

    @classmethod
    def m(cls):
        """the dimensions of the space"""
        return 3

    @classmethod
    def n(cls):
        """the dimensions of the element"""
        return 3

    @classmethod
    def _etype(cls):
        return 'unique msepy curvilinear hexahedron'

    @property
    def metric_signature(self):
        """return int when it is unique."""
        return id(self)

    @classmethod
    def face_setting(cls):
        """To show the nodes of faces and the positive direction."""
        return {
            0: (0, 2, 4, 6),   # face #0: x- face, Upper
            1: (1, 3, 5, 7),   # face #1: x+ face, Down
            2: (0, 1, 4, 5),   # face #2: y- face, Left
            3: (2, 3, 6, 7),   # face #3: y+ face, Right
            4: (0, 1, 2, 3),   # face #2: z- face, Back
            5: (4, 5, 6, 7),   # face #3: z+ face, Front
        }

    @property
    def faces(self):
        """The faces of this element."""
        if self._faces is None:
            self._faces = MseHtt_GreatMesh_Curvilinear_Hexahedron_Element_Faces(self)
        return self._faces

    def _generate_outline_data(self, ddf=1):
        """"""
        if ddf <= 0.1:
            ddf = 0.1
        else:
            pass
        samples = 17 * ddf
        if samples >= 35:
            samples = 35
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        linspace_0 = np.concatenate([
            np.linspace(-1, 1, samples),
            np.linspace(1, 1, samples),
            np.linspace(1, -1, samples),
            np.linspace(-1, -1, samples)
        ])
        linspace_1 = np.concatenate([
            np.linspace(-1, -1, samples),
            np.linspace(-1, 1, samples),
            np.linspace(1, 1, samples),
            np.linspace(1, -1, samples)
        ])
        p_ones = np.ones_like(linspace_0)
        m_ones = - p_ones

        return {
            'mn': (self.m(), self.n()),
            'center': self.ct.mapping(0, 0, 0),
            0: self.ct.mapping(m_ones, linspace_0, linspace_1),   # face #0, x-
            1: self.ct.mapping(p_ones, linspace_0, linspace_1),   # face #1, x+
            2: self.ct.mapping(linspace_0, m_ones, linspace_1),   # face #2, y-
            3: self.ct.mapping(linspace_0, p_ones, linspace_1),   # face #3, y+
            4: self.ct.mapping(linspace_0, linspace_1, m_ones),   # face #4, z-
            5: self.ct.mapping(linspace_0, linspace_1, p_ones),   # face #5, z+
        }


# ============ ELEMENT CT =====================================================================================
class MseHtt_GreatMesh_Unique_Msepy_Curvilinear_Hexahedron_Element_CooTrans(MseHttGreatMeshBaseElementCooTrans):
    """"""

    def mapping(self, xi, et, sg):
        """"""
        md_ref_coo = list()
        for j, _ in enumerate([xi, et, sg]):
            _ = (_ + 1) * 0.5 * self._element._delta[j] + self._element._origin[j]
            md_ref_coo.append(_)

        return self._element._msepy_manifold.ct.mapping(
            *md_ref_coo, regions=self._element._region
        )[self._element._region]

    def ___Jacobian_matrix___(self, xi, et, sg):
        """"""
        md_ref_coo = list()
        for j, _ in enumerate((xi, et, sg)):
            _ = (_ + 1) * 0.5 * self._element._delta[j] + self._element._origin[j]
            md_ref_coo.append(_)

        jm = self._element._msepy_manifold.ct.Jacobian_matrix(
            *md_ref_coo, regions=self._element._region
        )[self._element._region]

        JM = tuple([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        for i in range(3):
            for j in range(3):
                jm_ij = jm[i][j]
                jm_ij *= self._element._delta[j] * 0.5
                JM[i][j] = jm_ij
        return JM


# ============ FACES ============================================================================================
class MseHtt_GreatMesh_Curvilinear_Hexahedron_Element_Faces(Frozen):
    r""""""
    def __init__(self, element):
        """"""
        self._element = element
        self._faces = {}
        self._freeze()

    def __getitem__(self, face_id):
        r"""0, 1, 2, 3, 4, 5"""
        assert face_id in range(6), f"face id must be in range(4)."
        if face_id not in self._faces:
            self._faces[face_id] = MseHtt_GreatMesh_Curvilinear_Hexahedron_Element_Face(self._element, face_id)
        else:
            pass
        return self._faces[face_id]

    def __repr__(self):
        """"""
        return f"<Faces of {self._element}>"


class MseHtt_GreatMesh_Curvilinear_Hexahedron_Element_Face(Frozen):
    """"""
    def __init__(self, element, face_id):
        self._element = element
        self._id = face_id
        self._ct = MseHtt_GreatMeshUniqueMsepy_CurvilinearHexahedron_Element_Face_CT(self)
        self._area = None
        self._freeze()

    def __repr__(self):
        """"""
        return f"<Face#{self._id} of {self._element}>"

    @property
    def ct(self):
        """Coordinate transformation of this face."""
        return self._ct

    @property
    def area(self):
        """The area of this element face."""
        if self._area is None:
            if self.ct.is_perp_plane() and self.ct.is_rectangle():
                # the face is a rectangle perp to an axis
                nodes = np.array([-1, 1])
                x, y, z = self._ct.mapping(nodes, nodes, nodes)
                x0, x1 = x
                y0, y1 = y
                z0, z1 = z
                if np.isclose(x0, x1):
                    self._area = (y1 - y0) * (z1 - z0)
                elif np.isclose(y0, y1):
                    self._area = (z1 - z0) * (x1 - x0)
                elif np.isclose(z0, z1):
                    self._area = (x1 - x0) * (y1 - y0)
                else:
                    raise Exception()
            else:
                raise NotImplementedError()

        return self._area


from msehtt.static.mesh.great.elements.types.orthogonal_hexahedron import (
    MseHtt_GreatMesh_OrthogonalHexahedron_Element_OneFace_CT)


from tools.miscellaneous.geometries.m3n3 import Point3, angle3


class MseHtt_GreatMeshUniqueMsepy_CurvilinearHexahedron_Element_Face_CT(
    MseHtt_GreatMesh_OrthogonalHexahedron_Element_OneFace_CT
):
    r""""""
    def __init__(self, face):
        r""""""
        super().__init__(face)
        self._melt()
        self.___is_place___ = None
        self.___is_perp_plane___ = None
        self.___is_rectangle___ = None
        self._freeze()

    def is_plane(self):
        r""""""
        if self.___is_place___ is None:
            xi = np.linspace(-1, 1, 13)
            ounv = self.outward_unit_normal_vector(xi, xi, xi)
            n0, n1, n2 = ounv
            if np.allclose(n0, n0[0]) and np.allclose(n1, n1[0]) and np.allclose(n2, n2[0]):
                self.___is_place___ = True
            else:
                self.___is_place___ = False
        return self.___is_place___

    def is_perp_plane(self):
        r"""Whether the element is a plane perp to an axis?"""
        if self.___is_perp_plane___ is None:
            if self.is_plane():
                xi = np.linspace(-1, 1, 13)
                ounv = self.outward_unit_normal_vector(xi, xi, xi)
                n0, n1, n2 = ounv
                if np.allclose(n0, 0) or np.allclose(n1, 0) or np.allclose(n2, 0):
                    self.___is_perp_plane___ = True
                else:
                    self.___is_perp_plane___ = False
            else:
                self.___is_perp_plane___ = False
        return self.___is_perp_plane___

    def is_rectangle(self):
        r"""Whether the element face is a rectangle?"""
        if self.___is_rectangle___ is None:
            if self.is_plane():
                xi = np.array([-1, 1, 1, -1])
                et = np.array([-1, -1, 1, 1])
                m, n = self._axis, self._start_end
                if m == 0:  # face perp to x-axis
                    XYZ = self.mapping(0, xi, et)
                elif m == 1:
                    XYZ = self.mapping(xi, 0, et)
                elif m == 2:
                    XYZ = self.mapping(xi, et, 0)
                else:
                    raise Exception()
                X, Y, Z = XYZ
                A = Point3(X[0], Y[0], Z[0])
                B = Point3(X[1], Y[1], Z[1])
                C = Point3(X[2], Y[2], Z[2])
                D = Point3(X[3], Y[3], Z[3])
                angle_ABC = angle3(A, B, C)
                angle_CDA = angle3(C, D, A)
                angle_90_degree = np.pi * 0.5

                if np.isclose(angle_ABC, angle_90_degree) and np.isclose(angle_CDA, angle_90_degree):
                    # this is actually not enough, but mostly, it is okay.
                    self.___is_rectangle___ = True
                else:
                    self.___is_rectangle___ = False

            else:
                self.___is_rectangle___ = False

        return self.___is_rectangle___
