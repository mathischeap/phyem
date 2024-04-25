# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM
from msehtt.static.mesh.partial.boundary_section.visualize.main import MseHttBoundarySectionPartialMeshVisualize


class MseHttBoundarySectionPartialMesh(Frozen):
    """A bunch of element faces of the great mesh."""

    def __init__(self, tpm, tgm, local_faces_list_of_tuples___element_index__plus__face_id):
        """
        if ``local_faces_list_of_tuples___element_index__plus__face_id = [(0, 0), (15, 3), (1143, 2), ...]``,

            (0, 0) means the 0th face of element #0
            (1143, 2) means the 2nd face of element #1143
            ...

        """
        self._tpm = tpm
        self._tgm = tgm

        self._face_dict = {}  # this boundary section includes these local element faces.
        for element_index__face_id in local_faces_list_of_tuples___element_index__plus__face_id:
            element_index, face_id = element_index__face_id
            assert element_index in tgm.elements, f"element indexed {element_index} is not a valid local great element."
            element = tgm.elements[element_index]
            assert face_id in element.face_setting(), f"face id = {face_id} is not a valid face id."
            the_face_instance = element.faces[face_id]
            self._face_dict[(element_index, face_id)] = the_face_instance

        self._visualize = None
        self._mn = None
        self._freeze()

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._tpm._abstract._sym_repr + super_repr

    def __len__(self):
        """how many local element faces?"""
        return len(self._face_dict)

    def __contains__(self, element_index__face_id):
        """If the face indicated by (element_index, face_id) is included locally by this boundary section?"""
        return element_index__face_id in self._face_dict

    def __iter__(self):
        """go through all local indicators of included element faces."""
        for element_index__face_id in self._face_dict:
            yield element_index__face_id

    def __getitem__(self, element_index__face_id):
        """Return the face indicated by (element_index, face_id)."""
        return self._face_dict[element_index__face_id]

    @property
    def visualize(self):
        """visualization."""
        if self._visualize is None:
            self._visualize = MseHttBoundarySectionPartialMeshVisualize(self)
        return self._visualize

    @property
    def mn(self):
        """The `m` and `n` of the elements I am on. All the elements across all ranks will take into account.
        So if the elements in a local rank are of one (m, n), (m, n) could be different in other ranks. So
        self.mn will return a tuple if two pairs of (m, n).

        For example, if self.mn = (2, 2), then all the elements I am on are 2d (n=2) elements in 2d (m=2) space.

        If self.mn == ((3, 2), (2, 2)), then some of the elements I am on are 2d (n=2) elements in 3d (m=2) space,
        and some other elements are 2d (n=2) element in 2d (m=2) space. Of course, this is not very likely.
        """
        if self._mn is None:
            mn_pool = list()
            for element_index___face_id in self:
                element_index, face_id = element_index___face_id
                element = self._tgm.elements[element_index]
                mn = (element.m, element.n)
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
