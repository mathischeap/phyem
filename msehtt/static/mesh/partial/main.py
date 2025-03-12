# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.miscellaneous.geometries.m2n2 import Point2, StraightLine2, whether_point_on_straight_line
from tools.miscellaneous.geometries.m2n2 import Polygon2, whether_point_in_polygon
from tools.miscellaneous.geometries.m2n2 import Curve2, whether_point_on_curve
from tools.miscellaneous.geometries.m2n2 import StraightSegment2, whether_point_on_straight_segment

from tools.frozen import Frozen
from src.mesh import Mesh
from msehtt.static.mesh.partial.elements.main import MseHttElementsPartialMesh
from msehtt.static.mesh.partial.boundary_section.main import MseHttBoundarySectionPartialMesh


class EmptyCompositionError(Exception):
    """"""


class MseHttMeshPartial(Frozen):
    """"""

    def __init__(self, abstract_mesh):
        """"""
        assert abstract_mesh.__class__ is Mesh, f"I need an abstract mesh."
        self._abstract = abstract_mesh
        self._tgm = None
        self._composition = None
        self._freeze()

    def info(self):
        """info self."""
        try:
            composition = self.composition
        except EmptyCompositionError:
            print(f"Mesh not-configured: {self.abstract._sym_repr}.")
        else:
            composition.info()

    @property
    def ___is_msehtt_partial_mesh___(self):
        return True

    @property
    def abstract(self):
        """return the abstract mesh instance."""
        return self._abstract

    @property
    def tgm(self):
        """Raise Error if it is not set yet!"""
        if self._tgm is None:
            raise Exception('tgm is empty!')
        return self._tgm

    @property
    def composition(self):
        """The composition; the main body of this partial mesh."""
        if self._composition is None:
            raise EmptyCompositionError(f"msehtt partial mesh of {self.abstract._sym_repr} has no composition yet")
        else:
            return self._composition

    @property
    def cfl(self):
        """About the CFL condition."""
        return self.composition.cfl

    @property
    def rws(self):
        """About the rws-grouped data saving or exporting."""
        return self.composition.rws

    @property
    def compute(self):
        """Compute something (other than cfl number) on this partial mesh."""
        return self.composition.compute

    @property
    def visualize(self):
        """Call the visualization scheme of the composition."""
        return self.composition.visualize

    def find_dofs(self, f, local=True):
        r"""Find the dofs of f on this partial mesh. If local is True, we return the results element-wise."""
        return self.composition.find_dofs(f, local=local)

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr

    def _config(self, tgm, including):
        r""""""

        # all boundary section strings
        ___bs___ = (
            "boundary_section", "boundary-section", "boundary section"
        )

        assert self._tgm is None, f"tgm must be set."
        assert self._composition is None, f"components are not set!"
        self._tgm = tgm

        if including == 'all':
            # CONFIGURATION 1: -----------------------------------------------------------
            # this partial mesh includes all elements of the great mesh.
            including = {
                'type': 'local great elements',
                'range': self._tgm.elements._elements_dict.keys()
            }

            CONFIGURATION = 1

        elif including.__class__ is self.__class__:
            abstract = including.abstract
            if including.composition.__class__ is MseHttElementsPartialMesh and abstract.boundary() is self.abstract:
                # CONFIGURATION 2: ----------------------------------------------------------
                # this partial mesh to be the boundary of a partial mesh (``including``) of a bunch of great elements
                #
                # Here we are config a boundary section.
                including = {
                    'type': 'boundary of partial elements',
                    'partial elements': including
                }

                CONFIGURATION = 2

            else:
                raise NotImplementedError()

        elif isinstance(including, dict):
            keys = set(including.keys())
            if keys == {'type', 'partial elements', 'ounv'} and including['type'] in ___bs___:
                # CONFIGURATION 3: ----------------------------------------------------------
                # config the partial mesh to be boundary section of faces whose outward unit vectors
                # are among ``including['ouv']. These faces are boundary faces of ``including['partial elements']``.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,           # mesh is a partial mesh of elements composition.
                #       'ounv': ([-1, 0], )                 # outward unit norm vectors.
                #   }
                #
                # This configures a boundary section which includes all boundary faces of ``mesh`` (must be an
                # elements partial mesh) whose outward unit norm vectors are all ``[-1, 0]``.
                #
                # Here we are config a boundary section.
                ounv = including['ounv']
                if all([isinstance(_, (int, float)) for _ in ounv]):  # we get only one vector
                    including['ounv'] = (ounv, )
                else:
                    pass

                CONFIGURATION = 3

            elif keys == {'type', 'partial elements', 'except ounv'} and including['type'] in ___bs___:
                # CONFIGURATION 4: ----------------------------------------------------------
                # config the partial mesh to be boundary section of faces whose outward unit vectors
                # are NOT among ``including['ouv']. These faces are boundary faces of ``including['partial elements']``.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,           # mesh is a partial mesh of elements composition.
                #       'except ounv': ([-1, 0], )          # except outward unit norm vectors.
                #   }
                #
                # This configures a boundary section which includes all boundary faces of ``mesh`` (must be an
                # elements partial mesh) whose outward unit norm vectors are not ``[-1, 0]``.
                #
                # Here we are config a boundary section.
                ounv = including['except ounv']
                if all([isinstance(_, (int, float)) for _ in ounv]):  # we get only one vector
                    including['except ounv'] = (ounv, )
                else:
                    pass

                CONFIGURATION = 4

            elif keys == {'type', 'partial elements', 'on straight lines'} and including['type'] in ___bs___:
                # CONFIGURATION 5: ----------------------------------------------------------
                #
                # Only works for face of m2n2 elements. Those faces that are on the lines are collected.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,           # mesh is a partial mesh of elements composition.
                #       'on straight lines': ([p0, p1], )   # on these lines
                #   }
                #
                # Here we are config a boundary section.
                the_partial_elements = including['partial elements'].composition
                assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                    f"I need a {MseHttElementsPartialMesh}."
                m, n = the_partial_elements.mn
                if m == n == 2:
                    CONFIGURATION = '5=m2n2'
                else:
                    raise NotImplementedError()

            elif keys == {'type', 'partial elements', 'except on straight lines'} and including['type'] in ___bs___:
                # CONFIGURATION 6: ----------------------------------------------------------
                #
                # Only works for face of m2n2 elements. Those faces that are NOT on the lines are collected.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,                          # A partial mesh of elements composition.
                #       'except on straight lines': ([p0, p1], [p2, p3])   # except these lines
                #   }
                #
                # Here we are config a boundary section.
                the_partial_elements = including['partial elements'].composition
                assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                    f"I need a {MseHttElementsPartialMesh}."
                m, n = the_partial_elements.mn
                if m == n == 2:
                    CONFIGURATION = '6=m2n2'
                else:
                    raise NotImplementedError()

            elif keys == {'type', 'partial elements', 'in polygons'} and including['type'] in ___bs___:
                # CONFIGURATION 7: ----------------------------------------------------------
                #
                # Only works for face of m2n2 elements. Those faces that are in (or on edge of) polygon are collected.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,            # mesh is a partial mesh of elements composition.
                #       'in polygons': (polygon0, polygon1)  # in (or on edge of) these polygons
                #   }
                #
                # Here we are config a boundary section.

                CONFIGURATION = 7

            elif keys == {'type', 'partial elements', 'except in polygons'} and including['type'] in ___bs___:
                # CONFIGURATION 8: ----------------------------------------------------------
                #
                # Only works for face of m2n2 elements. Faces that are NOT in (or on edge of) polygon are collected.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,                   # mesh is a partial mesh of elements composition.
                #       'except in polygons': (polygon0, polygon1)  # NOT in (or on edge of) these polygons
                #   }
                #
                # Here we are config a boundary section.

                CONFIGURATION = 8

            elif keys == {'type', 'partial elements', 'on curves'} and including['type'] in ___bs___:
                # CONFIGURATION 9: ----------------------------------------------------------
                #
                # Only works for face of m2n2 elements. Those faces that are on these curves are collected.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,           # mesh is a partial mesh of elements composition.
                #       'on curves': (curve0, curve1)       # on these curves
                #   }
                #
                # Here we are config a boundary section.
                the_partial_elements = including['partial elements'].composition
                assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                    f"I need a {MseHttElementsPartialMesh}."
                m, n = the_partial_elements.mn
                if m == n == 2:
                    CONFIGURATION = '9=m2n2'
                else:
                    raise NotImplementedError()

            elif keys == {'type', 'partial elements', 'except on curves'} and including['type'] in ___bs___:
                # CONFIGURATION 10: ----------------------------------------------------------
                #
                # Only works for face of m2n2 elements. Faces that are NOT on these curves are collected.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,                  # mesh is a partial mesh of elements composition.
                #       'except on curves': (curve0, curve1)       # NOT on these curves
                #   }
                #
                # Here we are config a boundary section.
                the_partial_elements = including['partial elements'].composition
                assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                    f"I need a {MseHttElementsPartialMesh}."
                m, n = the_partial_elements.mn
                if m == n == 2:
                    CONFIGURATION = '10=m2n2'
                else:
                    raise NotImplementedError()

            elif keys == {'type', 'partial elements', 'on straight segments'} and including['type'] in ___bs___:
                # CONFIGURATION 11: ----------------------------------------------------------
                #
                # Only works for face of m2n2 elements. Those faces that are on the segments are collected.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,              # mesh is a partial mesh of elements composition.
                #       'on straight segments': ([p0, p1], )   # on these m2n2 segments
                #   }
                #
                # Here we are config a boundary section.
                the_partial_elements = including['partial elements'].composition
                assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                    f"I need a {MseHttElementsPartialMesh}."
                m, n = the_partial_elements.mn
                if m == n == 2:
                    CONFIGURATION = '11=m2n2'
                else:
                    raise NotImplementedError()

            elif keys == {'type', 'partial elements', 'except on straight segments'} and including['type'] in ___bs___:
                # CONFIGURATION 12: ----------------------------------------------------------
                #
                # Only works for face of m2n2 elements. Those faces that are NOT on the segments are collected.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,                             # A partial mesh of elements composition.
                #       'except on straight segments': ([p0, p1], [p2, p3])   # except these m2n2 straight segments
                #   }
                #
                # Here we are config a boundary section.
                the_partial_elements = including['partial elements'].composition
                assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                    f"I need a {MseHttElementsPartialMesh}."
                m, n = the_partial_elements.mn
                if m == n == 2:
                    CONFIGURATION = '12=m2n2'
                else:
                    raise NotImplementedError()

            else:
                raise Exception(f"Parse no configuration from {including}.")

        else:
            raise NotImplementedError()

        self._perform_configuration(including, CONFIGURATION)

    def _perform_configuration(self, including, CONFIGURATION):
        r"""Really do it according to the configuration."""
        if CONFIGURATION == 1:
            # CONFIGURATION 1 ===========================================================================
            # this partial mesh consists of local (rank) elements of the great mesh.
            rank_great_element_range = including['range']  # the range of local great elements.
            self._composition = MseHttElementsPartialMesh(self, self._tgm, rank_great_element_range)

        elif CONFIGURATION == 2:
            # CONFIGURATION 2 ===========================================================================
            # this partial mesh consists of boundary faces of a bunch of local partial elements.
            the_partial_elements = including['partial elements'].composition  # the local partial elements
            # now we try to get indices of the included faces.
            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, local_boundary_faces_information)

        elif CONFIGURATION == 3:
            # CONFIGURATION 3 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            outward_unit_norm_vectors = including['ounv']

            OUTWARD_UNIT_NORM_VECTORS = list()
            for vector in outward_unit_norm_vectors:
                round_vector = list()
                for digit in vector:
                    round_vector.append(round(digit, 6))
                OUTWARD_UNIT_NORM_VECTORS.append(tuple(round_vector))

            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            valid_rank_faces = list()
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                if face.ct.is_plane():
                    c_ounv = face.ct.constant_outward_unit_normal_vector
                    if c_ounv in OUTWARD_UNIT_NORM_VECTORS:
                        # this face is a valid boundary section face.
                        valid_rank_faces.append(rank___element_index__face_id)
                    else:
                        pass
                else:
                    pass
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        elif CONFIGURATION == 4:
            # CONFIGURATION 4 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            outward_unit_norm_vectors = including['except ounv']

            OUTWARD_UNIT_NORM_VECTORS = list()
            for vector in outward_unit_norm_vectors:
                round_vector = list()
                for digit in vector:
                    round_vector.append(round(digit, 6))
                OUTWARD_UNIT_NORM_VECTORS.append(tuple(round_vector))

            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            valid_rank_faces = list()
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                if face.ct.is_plane():
                    c_ounv = face.ct.constant_outward_unit_normal_vector
                    if c_ounv in OUTWARD_UNIT_NORM_VECTORS:
                        pass
                    else:
                        # this face is a valid boundary section face.
                        valid_rank_faces.append(rank___element_index__face_id)
                else:
                    valid_rank_faces.append(rank___element_index__face_id)
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        elif CONFIGURATION == '5=m2n2':
            # CONFIGURATION 5 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            m, n = the_partial_elements.mn
            assert m == n == 2, f"The partial mesh must be 2d in a 2d space."

            valid_lines = including['on straight lines']
            # the faces (of 2d elements) on these lines are the faces we are looking for.

            VALID_LINES = list()
            for vl in valid_lines:
                if (isinstance(vl, (list, tuple)) and
                        len(vl) == 2 and
                        vl[0].__class__ is Point2 and
                        vl[1].__class__ is Point2):
                    vl = StraightLine2(vl[0], vl[1])
                elif (isinstance(vl, (list, tuple)) and
                        len(vl) == 2 and
                        isinstance(vl[0], (tuple, list)) and
                        len(vl[0]) == 2 and
                        isinstance(vl[0][0], (float, int)) and
                        isinstance(vl[0][1], (float, int)) and
                        isinstance(vl[1], (tuple, list)) and
                        len(vl[1]) == 2 and
                        isinstance(vl[1][0], (float, int)) and
                        isinstance(vl[1][1], (float, int))):  # for example, vl = [(a, b), (c, d)]
                    vl = StraightLine2(Point2(*vl[0]), Point2(*vl[1]))
                else:
                    assert vl.__class__ is StraightLine2, f"it must be a m2n2 StraightLine2 instance."
                VALID_LINES.append(vl)

            for VL in VALID_LINES:
                assert isinstance(VL, StraightLine2), f"just to check."

            valid_rank_faces = list()
            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                for line_instance in VALID_LINES:
                    OnLine_Or_Not = ___checking_whether_a_face_of_a_m2n2_element_is_on_straight_line___(
                        face, line_instance)
                    if OnLine_Or_Not:
                        valid_rank_faces.append(rank___element_index__face_id)
                        break
                    else:
                        pass
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        elif CONFIGURATION == '6=m2n2':
            # CONFIGURATION 6 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            m, n = the_partial_elements.mn
            assert m == n == 2, f"The partial mesh must be 2d in a 2d space."

            valid_lines = including['except on straight lines']
            # the faces (of 2d elements) on these lines are the faces we are looking for.

            VALID_LINES = list()
            for vl in valid_lines:
                if (isinstance(vl, (list, tuple)) and
                        len(vl) == 2 and
                        vl[0].__class__ is Point2 and
                        vl[1].__class__ is Point2):
                    vl = StraightLine2(vl[0], vl[1])
                elif (isinstance(vl, (list, tuple)) and
                        len(vl) == 2 and
                        isinstance(vl[0], (tuple, list)) and
                        len(vl[0]) == 2 and
                        isinstance(vl[0][0], (float, int)) and
                        isinstance(vl[0][1], (float, int)) and
                        isinstance(vl[1], (tuple, list)) and
                        len(vl[1]) == 2 and
                        isinstance(vl[1][0], (float, int)) and
                        isinstance(vl[1][1], (float, int))):  # for example, vl = [(a, b), (c, d)]
                    vl = StraightLine2(Point2(*vl[0]), Point2(*vl[1]))
                else:
                    assert vl.__class__ is StraightLine2, f"it must be a m2n2 StraightLine2 instance."
                VALID_LINES.append(vl)

            valid_rank_faces = list()
            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                ToF = False
                for line_instance in VALID_LINES:
                    OnLine_Or_Not = ___checking_whether_a_face_of_a_m2n2_element_is_on_straight_line___(
                        face, line_instance)
                    if OnLine_Or_Not:
                        ToF = True
                        break
                    else:
                        pass
                if ToF:
                    pass
                else:
                    valid_rank_faces.append(rank___element_index__face_id)
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        elif CONFIGURATION == 7:
            # CONFIGURATION 7 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            m, n = the_partial_elements.mn
            assert m == n == 2, f"The partial mesh must be 2d in a 2d space."

            valid_polygons = including['in polygons']
            # the faces (of 2d elements) on these lines are the faces we are looking for.

            VALID_POLYGONS = list()
            if isinstance(valid_polygons, Polygon2):
                VALID_POLYGONS.append(valid_polygons)
            else:
                assert isinstance(valid_polygons, (list, tuple)), f"pls put polygons in a list or tuple."
                for vp in valid_polygons:
                    if isinstance(vp, Polygon2):
                        VALID_POLYGONS.append(vp)
                    else:
                        raise NotImplementedError(f"I do not understand {vp}")

            valid_rank_faces = list()
            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                for polygon in VALID_POLYGONS:
                    OnLine_Or_Not = ___checking_whether_a_face_of_a_m2n2_element_is_in_a_polygon___(
                        face, polygon)
                    if OnLine_Or_Not:
                        valid_rank_faces.append(rank___element_index__face_id)
                        break
                    else:
                        pass
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        elif CONFIGURATION == 8:
            # CONFIGURATION 8 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            m, n = the_partial_elements.mn
            assert m == n == 2, f"The partial mesh must be 2d in a 2d space."

            valid_polygons = including['except in polygons']
            # the faces (of 2d elements) on these lines are the faces we are looking for.

            VALID_POLYGONS = list()
            if isinstance(valid_polygons, Polygon2):
                VALID_POLYGONS.append(valid_polygons)
            else:
                assert isinstance(valid_polygons, (list, tuple)), f"pls put polygons in a list or tuple."
                for vp in valid_polygons:
                    if isinstance(vp, Polygon2):
                        VALID_POLYGONS.append(vp)
                    else:
                        raise NotImplementedError(f"I do not understand {vp}")

            valid_rank_faces = list()
            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                ToF = False
                for polygon in VALID_POLYGONS:
                    OnLine_Or_Not = ___checking_whether_a_face_of_a_m2n2_element_is_in_a_polygon___(
                        face, polygon)
                    if OnLine_Or_Not:
                        ToF = True
                        break
                    else:
                        pass
                if ToF:
                    pass
                else:
                    valid_rank_faces.append(rank___element_index__face_id)
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        elif CONFIGURATION == '9=m2n2':
            # CONFIGURATION 9 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            m, n = the_partial_elements.mn
            assert m == n == 2, f"The partial mesh must be 2d in a 2d space."

            valid_curves = including['on curves']
            # the faces (of 2d elements) on these lines are the faces we are looking for.

            VALID_CURVES = list()
            for vc in valid_curves:
                assert vc.__class__ is Curve2, f"it must be a m2n2 StraightLine2 instance."
                VALID_CURVES.append(vc)

            for VC in VALID_CURVES:
                assert isinstance(VC, Curve2), f"just to check."

            valid_rank_faces = list()
            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                for curve in VALID_CURVES:
                    OnCurve_Or_Not = ___checking_whether_a_face_of_a_m2n2_element_is_on_curve___(
                        face, curve)
                    if OnCurve_Or_Not:
                        valid_rank_faces.append(rank___element_index__face_id)
                        break
                    else:
                        pass
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        elif CONFIGURATION == '10=m2n2':
            # CONFIGURATION 10 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            m, n = the_partial_elements.mn
            assert m == n == 2, f"The partial mesh must be 2d in a 2d space."

            valid_curves = including['except on curves']
            # the faces (of 2d elements) on these lines are the faces we are looking for.

            VALID_CURVES = list()
            for vc in valid_curves:
                assert vc.__class__ is Curve2, f"it must be a m2n2 StraightLine2 instance."
                VALID_CURVES.append(vc)

            valid_rank_faces = list()
            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                ToF = False
                for curve in VALID_CURVES:
                    OnCurve_Or_Not = ___checking_whether_a_face_of_a_m2n2_element_is_on_curve___(
                        face, curve)
                    if OnCurve_Or_Not:
                        ToF = True
                        break
                    else:
                        pass
                if ToF:
                    pass
                else:
                    valid_rank_faces.append(rank___element_index__face_id)
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        elif CONFIGURATION == '11=m2n2':
            # CONFIGURATION 5 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            m, n = the_partial_elements.mn
            assert m == n == 2, f"The partial mesh must be 2d in a 2d space."

            valid_segments = including['on straight segments']
            # the faces (of 2d elements) on these lines are the faces we are looking for.

            VALID_SEGMENTS = list()
            for vl in valid_segments:
                if (isinstance(vl, (list, tuple)) and
                        len(vl) == 2 and
                        vl[0].__class__ is Point2 and
                        vl[1].__class__ is Point2):
                    vl = StraightSegment2(vl[0], vl[1])
                elif (isinstance(vl, (list, tuple)) and
                        len(vl) == 2 and
                        isinstance(vl[0], (tuple, list)) and
                        len(vl[0]) == 2 and
                        isinstance(vl[0][0], (float, int)) and
                        isinstance(vl[0][1], (float, int)) and
                        isinstance(vl[1], (tuple, list)) and
                        len(vl[1]) == 2 and
                        isinstance(vl[1][0], (float, int)) and
                        isinstance(vl[1][1], (float, int))):  # for example, vl = [(a, b), (c, d)]
                    vl = StraightSegment2(Point2(*vl[0]), Point2(*vl[1]))
                else:
                    assert vl.__class__ is StraightSegment2, f"it must be a m2n2 StraightSegment2 instance."
                VALID_SEGMENTS.append(vl)

            for VL in VALID_SEGMENTS:
                assert isinstance(VL, StraightSegment2), f"just to check."

            valid_rank_faces = list()
            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                for segment in VALID_SEGMENTS:
                    OnLine_Or_Not = ___checking_whether_a_face_of_a_m2n2_element_is_on_straight_segment___(
                        face, segment)
                    if OnLine_Or_Not:
                        valid_rank_faces.append(rank___element_index__face_id)
                        break
                    else:
                        pass
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        elif CONFIGURATION == '12=m2n2':
            # CONFIGURATION 6 ===========================================================================
            the_partial_elements = including['partial elements'].composition
            assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                f"I need a {MseHttElementsPartialMesh}."
            m, n = the_partial_elements.mn
            assert m == n == 2, f"The partial mesh must be 2d in a 2d space."

            valid_segments = including['except on straight segments']
            # the faces (of 2d elements) on these lines are the faces we are looking for.

            VALID_SEGMENTS = list()
            for vl in valid_segments:
                if (isinstance(vl, (list, tuple)) and
                        len(vl) == 2 and
                        vl[0].__class__ is Point2 and
                        vl[1].__class__ is Point2):
                    vl = StraightSegment2(vl[0], vl[1])
                elif (isinstance(vl, (list, tuple)) and
                        len(vl) == 2 and
                        isinstance(vl[0], (tuple, list)) and
                        len(vl[0]) == 2 and
                        isinstance(vl[0][0], (float, int)) and
                        isinstance(vl[0][1], (float, int)) and
                        isinstance(vl[1], (tuple, list)) and
                        len(vl[1]) == 2 and
                        isinstance(vl[1][0], (float, int)) and
                        isinstance(vl[1][1], (float, int))):  # for example, vl = [(a, b), (c, d)]
                    vl = StraightSegment2(Point2(*vl[0]), Point2(*vl[1]))
                else:
                    assert vl.__class__ is StraightSegment2, f"it must be a m2n2 StraightSegment2 instance."
                VALID_SEGMENTS.append(vl)

            valid_rank_faces = list()
            local_boundary_faces_information = the_partial_elements._get_local_boundary_faces()
            # local_boundary_faces_information = [(0, 0), (15, 3), (1143, 2), ...]
            # (0, 0) means the 0th face of element #0
            # (1143, 2) means the 2nd face of element #1143
            # ...
            for rank___element_index__face_id in local_boundary_faces_information:
                element_index, face_id = rank___element_index__face_id
                face = self._tgm.elements[element_index].faces[face_id]
                ToF = False
                for segment in VALID_SEGMENTS:
                    OnLine_Or_Not = ___checking_whether_a_face_of_a_m2n2_element_is_on_straight_segment___(
                        face, segment)
                    if OnLine_Or_Not:
                        ToF = True
                        break
                    else:
                        pass
                if ToF:
                    pass
                else:
                    valid_rank_faces.append(rank___element_index__face_id)
            self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

        else:
            raise NotImplementedError(f"indicator={including} not understandable.")


def ___checking_whether_a_face_of_a_m2n2_element_is_on_straight_line___(face, line_instance):
    r"""

    Parameters
    ----------
    face :
        The face instance.
    line_instance :

    Returns
    -------

    """
    assert isinstance(line_instance, StraightLine2), f"straight line must be a {StraightLine2} instance."
    xi_or_et = np.linspace(-1, 1, 13)
    X, Y = face.ct.mapping(xi_or_et)
    On_or_Not = True
    for x, y in zip(X, Y):
        p = Point2(x, y)
        if whether_point_on_straight_line(p, line_instance):
            pass
        else:
            On_or_Not = False
            break
    return On_or_Not


def ___checking_whether_a_face_of_a_m2n2_element_is_in_a_polygon___(face, polygon):
    r""""""
    assert isinstance(polygon, Polygon2), f"polygon must be a {Polygon2} instance."
    xi_or_et = np.linspace(-1, 1, 13)
    X, Y = face.ct.mapping(xi_or_et)
    On_or_Not = True
    for x, y in zip(X, Y):
        p = Point2(x, y)
        if whether_point_in_polygon(p, polygon):
            pass
        else:
            On_or_Not = False
            break
    return On_or_Not


def ___checking_whether_a_face_of_a_m2n2_element_is_on_curve___(face, curve):
    r""""""
    assert isinstance(curve, Curve2), f"curve must be a {Curve2} instance."
    xi_or_et = np.linspace(-1, 1, 13)
    X, Y = face.ct.mapping(xi_or_et)
    On_or_Not = True
    for x, y in zip(X, Y):
        p = Point2(x, y)
        if whether_point_on_curve(p, curve):
            pass
        else:
            On_or_Not = False
            break
    return On_or_Not


def ___checking_whether_a_face_of_a_m2n2_element_is_on_straight_segment___(face, segment):
    r"""

    Parameters
    ----------
    face :
        The face instance.
    segment :

    Returns
    -------

    """
    assert isinstance(segment, StraightSegment2), f"straight segment must be a {StraightSegment2} instance."
    xi_or_et = np.linspace(-1, 1, 13)
    X, Y = face.ct.mapping(xi_or_et)
    On_or_Not = True
    for x, y in zip(X, Y):
        p = Point2(x, y)
        if whether_point_on_straight_segment(p, segment):
            pass
        else:
            On_or_Not = False
            break
    return On_or_Not
