# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen
from msehtt.static.mesh.partial.elements.visualize.main import MseHttElementsPartialMeshVisualize
from src.config import RANK, MASTER_RANK, COMM, SIZE

from msehtt.static.mesh.partial.elements.cfl import MseHtt_PartialMesh_Elements_CFL_condition
from msehtt.static.mesh.partial.elements.rws import MseHtt_PartialMesh_Elements_ExportTo_DDS_RWS_Grouped
from msehtt.static.mesh.partial.elements.compute import PartialMesh_Elements_Compute


class MseHttElementsPartialMesh(Frozen):
    """The partial mesh that contains only great elements (all or partial)."""

    def __init__(self, tpm, tgm, local_element_range):
        """"""
        self._tpm = tpm
        self._tgm = tgm
        for i in local_element_range:
            assert i in tgm.elements, f"element #{i} is not a valid local great mesh element."
        self._range = local_element_range
        self._visualize = None
        self._merge_element_indices()
        self._mn = None
        self._cfl = None
        self._rws = None
        self._compute = None

        self.___outline_plot_info___ = False  # do not use None; will only save in MASTER, save None in SLAVES.
        self._outline_plot_info()

        self._freeze()

    def info(self, additional_info=''):
        """info self."""
        print(
            f"{additional_info}"
            f"msehtt-partial-elements {self._tpm.abstract._sym_repr}: "
            f"{self._num_global_elements} elements"
        )

    @property
    def ___is_msehtt_partial_elements_mesh___(self):
        r"""Just a signature"""
        return True

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._tpm._abstract._sym_repr + super_repr

    @property
    def cfl(self):
        """We can study the cfl number on this mesh for a form."""
        if self._cfl is None:
            self._cfl = MseHtt_PartialMesh_Elements_CFL_condition(self)
        return self._cfl

    @property
    def rws(self):
        """We can export objects to a dds-rws-grouped instance based on this partial elements (a mesh basically)."""
        if self._rws is None:
            self._rws = MseHtt_PartialMesh_Elements_ExportTo_DDS_RWS_Grouped(self)
        return self._rws

    @property
    def compute(self):
        r"""Compute something (other than cfl number) based on this partial mesh of elements."""
        if self._compute is None:
            self._compute = PartialMesh_Elements_Compute(self)
        return self._compute

    @property
    def visualize(self):
        """To visualize this partial mesh of elements."""
        if self._visualize is None:
            self._visualize = MseHttElementsPartialMeshVisualize(self)
        return self._visualize

    def __iter__(self):
        """Go through all local valid element indices."""
        for i in self._range:
            yield i

    def __contains__(self, item):
        """if item is a valid local element index?"""
        return item in self._range

    def __len__(self):
        """How many local valid elements?"""
        return len(self._range)

    def __getitem__(self, item):
        """Return the local element indexed `item`."""
        assert item in self, f"element index {item} is not included in this partial element mesh."
        return self._tgm.elements[item]

    def _merge_element_indices(self):
        """"""
        all_ranges = COMM.gather(list(self._range), root=MASTER_RANK)
        if RANK == MASTER_RANK:
            self._global_element_indices = list()
            for _ in all_ranges:
                self._global_element_indices.extend(_)

            num_global_elements = len(self._global_element_indices)

            if all([isinstance(_, (int, float)) for _ in self._global_element_indices]):
                min_index = min(self._global_element_indices)
                max_index = max(self._global_element_indices)
                if set(self._global_element_indices) == set(range(min_index, max_index+1)):
                    self._global_element_indices = range(min_index, max_index+1)
                else:
                    pass
            else:
                pass

        else:
            num_global_elements = None

        self._num_global_elements = COMM.bcast(num_global_elements, root=MASTER_RANK)

    @property
    def global_element_range(self):
        """Return all the indices of the element across all ranks in the master rank, and return None in other ranks."""
        if RANK == MASTER_RANK:
            return self._global_element_indices
        else:
            return None

    @property
    def num_global_elements(self):
        """The amount of great elements this partial elements mesh has across all ranks."""
        return self._num_global_elements

    def _get_local_boundary_faces(self):
        """"""
        if RANK != MASTER_RANK:
            all_boundary_faces_information = None
        else:
            all_boundary_faces_information = self._tgm.elements._parse_boundary_faces_of_a_patch_of_elements(
                self.global_element_range
            )
        all_boundary_faces_information = COMM.bcast(all_boundary_faces_information, root=MASTER_RANK)

        local_boundary_faces_information = list()
        for boundary_face_information in all_boundary_faces_information:
            element_index = boundary_face_information['element index']
            if element_index in self:
                local_boundary_faces_information.append(
                    (element_index, boundary_face_information['face id'])
                )

        return local_boundary_faces_information

    @property
    def mn(self):
        """The `m` and `n` of the elements I am on. All the elements across all ranks will take into account.
        So if the elements in a local rank are of one (m, n), (m, n) could be different in other ranks. So
        self.mn will return a tuple if two pairs of (m, n).

        For example, if self.mn = (2, 2), then all the elements I am on are 2d (n=2) elements in 2d (m=2) space.

        If self.mn == ((3, 2), (2, 2)), then some of the elements I am on are 2d (n=2) elements in 3d (m=3) space,
        and some other elements are 2d (n=2) element in 2d (m=2) space. Of course, this is not very likely.
        """
        if self._mn is None:
            mn_pool = list()
            for element_index in self:
                element = self[element_index]
                mn = (element.m(), element.n())
                if mn not in mn_pool:
                    mn_pool.append(mn)
                else:
                    pass
            mn_pool = COMM.gather(mn_pool, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                total_mn_pool = set()
                for _ in mn_pool:
                    total_mn_pool.update(_)
                total_mn_pool = list(total_mn_pool)

                if len(total_mn_pool) == 1:
                    self._mn = total_mn_pool[0]
                else:
                    self._mn = tuple(total_mn_pool)
            else:
                self._mn = None

            self._mn = COMM.bcast(self._mn, root=MASTER_RANK)
        return self._mn

    def _outline_plot_info(self):
        r""""""
        if self.___outline_plot_info___ is False:

            if RANK == MASTER_RANK:
                all_boundary_faces_information = self._tgm.elements._parse_boundary_faces_of_a_patch_of_elements(
                    self.global_element_range)
            else:
                all_boundary_faces_information = None

            all_boundary_faces_information = COMM.bcast(all_boundary_faces_information, root=MASTER_RANK)

            local_boundary_faces_information = list()
            for boundary_face_information in all_boundary_faces_information:
                element_index = boundary_face_information['element index']
                if element_index in self:
                    local_boundary_faces_information.append(
                        (element_index, boundary_face_information['face id'])
                    )

            face_outline_linspace = {
                9: (np.array([-1, 1]), ),
                5: (np.array([-1, 1]), ),
            }

            ___outline_plot_info___ = {}

            for element_index__face_index in local_boundary_faces_information:
                element_index, face_index = element_index__face_index
                element = self[element_index]
                etype = element.etype
                if etype in face_outline_linspace:
                    linspace = face_outline_linspace[etype]
                    face_coo = element.faces[face_index].ct.mapping(*linspace)
                else:
                    face_coo = None  # to be implemented.

                ___outline_plot_info___[(etype, element_index, face_index)] = face_coo

            ___outline_plot_info___ = COMM.gather(___outline_plot_info___, root=MASTER_RANK)

            if RANK == MASTER_RANK:
                self.___outline_plot_info___ = {}
                for DICT in ___outline_plot_info___:
                    self.___outline_plot_info___.update(DICT)
            else:
                self.___outline_plot_info___ = None

        return self.___outline_plot_info___

    def _find_in_which_elements_(self, *args):
        r"""
        If the object is in or on multiple elements, we return all indices of these elements.

        """
        if all([isinstance(_, (int, float)) for _ in args]):  # all args are integers or floats.
            # This means we are receiving a coordinate.
            coo = args
            return self.___find_coordinates_in_which_elements___(*coo)
        else:
            raise NotImplementedError()

    def ___find_coordinates_in_which_elements___(self, *coordinates):
        r"""If the point is on the interface of multiple elements, we return all indices of these elements.

        We return the same outputs in all ranks. So, if the element is not a local element, we return its index as well.
        """
        mn = self.mn
        if mn == (2, 2):
            x, y =coordinates
            in_element = None
            for element_index in self:
                element = self[element_index]
                in_or_not = element._whether_coo_in_me_(x, y)
                if in_or_not:
                    in_element = element_index
                    break
                else:
                    pass

            in_element = COMM.gather(in_element, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                IN_ELEMENT = []
                for _e in in_element:
                    if _e is None:
                        pass
                    else:
                        IN_ELEMENT.append(_e)
            else:
                IN_ELEMENT = None

            return COMM.bcast(IN_ELEMENT, root=MASTER_RANK)

        else:
            raise NotImplementedError()
