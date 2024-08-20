# -*- coding: utf-8 -*-
r"""
"""
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

        elif including.__class__ is self.__class__:
            abstract = including.abstract
            if including.composition.__class__ is MseHttElementsPartialMesh and abstract.boundary() is self.abstract:
                # CONFIGURATION 2: ----------------------------------------------------------
                # this partial mesh to be the boundary of a partial mesh (``including``) of a bunch of great elements
                including = {
                    'type': 'boundary of partial elements',
                    'partial elements': including
                }

            else:
                raise NotImplementedError()

        elif isinstance(including, dict):
            keys = set(including.keys())
            if keys == {'type', 'partial elements', 'ounv'} and including['type'] == 'boundary_section':
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
                ounv = including['ounv']
                if all([isinstance(_, (int, float)) for _ in ounv]):  # we get only one vector
                    including['ounv'] = (ounv, )
                else:
                    pass

            elif keys == {'type', 'partial elements', 'except ounv'} and including['type'] == 'boundary_section':
                # CONFIGURATION 4: ----------------------------------------------------------
                # config the partial mesh to be boundary section of faces whose outward unit vectors
                # are NOT among ``including['ouv']. These faces are boundary faces of ``including['partial elements']``.
                #
                # For example:
                #
                #  including={
                #       'type': 'boundary_section',
                #       'partial elements': mesh,           # mesh is a partial mesh of elements composition.
                #       'except ounv': ([-1, 0], )                 # outward unit norm vectors.
                #   }
                #
                # This configures a boundary section which includes all boundary faces of ``mesh`` (must be an
                # elements partial mesh) whose outward unit norm vectors are not ``[-1, 0]``.
                ounv = including['except ounv']
                if all([isinstance(_, (int, float)) for _ in ounv]):  # we get only one vector
                    including['except ounv'] = (ounv, )
                else:
                    pass

            else:
                raise Exception()

        else:
            raise NotImplementedError()

        self._perform_configuration(including)

    def _perform_configuration(self, including):
        r"""Really do the configuration."""
        _type = including['type']
        if _type == 'local great elements':
            # CONFIGURATION 1 ===========================================================================
            # this partial mesh consists of local (rank) elements of the great mesh.
            rank_great_element_range = including['range']  # the range of local great elements.
            self._composition = MseHttElementsPartialMesh(self, self._tgm, rank_great_element_range)

        elif _type == 'boundary of partial elements':
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
        elif isinstance(including, dict):
            keys = set(including.keys())
            if keys == {'type', 'partial elements', 'ounv'} and including['type'] == 'boundary_section':
                # CONFIGURATION 3 ===========================================================================
                the_partial_elements = including['partial elements'].composition
                assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                    f"I need a {MseHttElementsPartialMesh}."
                outward_unit_norm_vectors = including['ounv']
                outward_unit_norm_vectors = [tuple(_) for _ in outward_unit_norm_vectors]
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
                        if c_ounv in outward_unit_norm_vectors:
                            # this face is a valid boundary section face.
                            valid_rank_faces.append(rank___element_index__face_id)
                        else:
                            pass
                    else:
                        pass
                self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

            elif keys == {'type', 'partial elements', 'except ounv'} and including['type'] == 'boundary_section':
                # CONFIGURATION 4 ===========================================================================
                the_partial_elements = including['partial elements'].composition
                assert the_partial_elements.__class__ is MseHttElementsPartialMesh, \
                    f"I need a {MseHttElementsPartialMesh}."
                outward_unit_norm_vectors = including['except ounv']
                outward_unit_norm_vectors = [tuple(_) for _ in outward_unit_norm_vectors]
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
                        if c_ounv in outward_unit_norm_vectors:
                            pass
                        else:
                            # this face is a valid boundary section face.
                            valid_rank_faces.append(rank___element_index__face_id)
                    else:
                        valid_rank_faces.append(rank___element_index__face_id)
                self._composition = MseHttBoundarySectionPartialMesh(self, self._tgm, valid_rank_faces)

            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
