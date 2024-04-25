# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehtt.static.mesh.partial.elements.visualize.main import MseHttElementsPartialMeshVisualize
from src.config import RANK, MASTER_RANK, COMM


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
        self._freeze()

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._tpm._abstract._sym_repr + super_repr

    @property
    def visualize(self):
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
