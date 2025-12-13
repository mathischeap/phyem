# -*- coding: utf-8 -*-
r"""Like the unique msepy curvilinear quadrilateral element, but this element does not depend
on structured msepy regions.
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement


class UniqueCurvilinearQuad(MseHttGreatMeshBaseElement):
    r"""
    The real element is mapped from the following reference element.

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
        r""""""
        self._mp = parameters['mapping']             # [-1, 1]^2 into element in physical domain
        self._JM = parameters['Jacobian_matrix']     # [-1, 1]^2 into element in physical domain
        super().__init__()
        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._ct = UCQ_CT(self)

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<U-C-Q element indexed:{self._index}" + super_repr

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
        return 'unique curvilinear quad'

    @classmethod
    def _find_element_center_coo(cls, parameters):
        r""""""
        mp = parameters['mapping']
        x, y = mp(0, 0)
        return np.array([x, y])

    @classmethod
    def _find_mapping_(cls, parameters, x, y):
        r""""""
        return parameters['mapping'](x, y)

    @property
    def metric_signature(self):
        """return int when it is unique."""
        return id(self)

    def _generate_outline_data(self, ddf=1, internal_grid=0):
        """"""
        if ddf <= 0.1:
            ddf = 0.1
        else:
            pass
        samples = 30 * ddf
        if samples >= 100:
            samples = 100
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        linspace = np.linspace(-1, 1, samples)
        ones = np.ones_like(linspace)

        line_dict = {
            'mn': (self.m(), self.n()),
            'center': self.ct.mapping(0, 0),
            0: self.ct.mapping(-ones, linspace),   # face #0: North
            1: self.ct.mapping(ones, linspace),    # face #1: South
            2: self.ct.mapping(linspace, -ones),   # face #2: West
            3: self.ct.mapping(linspace, ones),    # face #3: East
        }

        if internal_grid == 0:
            return line_dict
        else:
            raise NotImplementedError()

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
            self._faces = UCQ_Faces(self)
        return self._faces

    def ___face_representative_str___(self):
        r""""""
        x = np.array([-1, 1, 0, 0])
        y = np.array([0, 0, -1, 1])
        x, y = self.ct.mapping(x, y)
        return {
            0: r"%.7f-%.7f" % (round(x[0], 7), round(y[0], 7)),
            1: r"%.7f-%.7f" % (round(x[1], 7), round(y[1], 7)),
            2: r"%.7f-%.7f" % (round(x[2], 7), round(y[2], 7)),
            3: r"%.7f-%.7f" % (round(x[3], 7), round(y[3], 7)),
        }

    @property
    def edges(self):
        raise Exception(f"U-C-Q (2d) element has no edges.")

    def ___edge_representative_str___(self):
        r""""""
        raise Exception(f"U-C-Q (2d) element has no edges.")


from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElementCooTrans


class UCQ_CT(MseHttGreatMeshBaseElementCooTrans):
    r""""""

    def __init__(self, ucq):
        """"""
        super().__init__(ucq, ucq.metric_signature)

    def mapping(self, xi, et):
        """"""
        return self._element._mp(xi, et)

    def ___Jacobian_matrix___(self, xi, et):
        """"""
        return self._element._JM(xi, et)


# ============ FACES ============================================================================================


class UCQ_Faces(Frozen):
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
            self._faces[face_id] = UCQ_Face(self._element, face_id)
        else:
            pass
        return self._faces[face_id]

    def __repr__(self):
        """"""
        return f"<Faces of {self._element}>"


class UCQ_Face(Frozen):
    """"""
    def __init__(self, element, face_id):
        self._element = element
        self._id = face_id
        self._ct = UCQ_Face_CT(self)
        self._freeze()

    def __repr__(self):
        """"""
        return f"<Face#{self._id} of {self._element}>"

    @property
    def ct(self):
        """Coordinate transformation of this face."""
        return self._ct


from phyem.msehtt.static.mesh.great.elements.types.orthogonal_rectangle import \
    MseHttGreatMeshOrthogonalRectangleElementFaceCT


class UCQ_Face_CT(
    MseHttGreatMeshOrthogonalRectangleElementFaceCT
):
    r""""""
    def __init__(self, face):
        r""""""
        super().__init__(face)
        self._melt()
        self.___is_place___ = None
        self._freeze()

    def is_plane(self):
        r""""""
        if self.___is_place___ is None:
            xi = np.linspace(-1, 1, 23)
            ounv = self.outward_unit_normal_vector(xi)
            n0, n1 = ounv
            if np.allclose(n0, n0[0]) and np.allclose(n1, n1[0]):
                self.___is_place___ = True
            else:
                self.___is_place___ = False
        return self.___is_place___
