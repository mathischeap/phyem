# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM, MPI, SIZE
from msehtt.static.mesh.partial.boundary_section.visualize.main import MseHttBoundarySectionPartialMeshVisualize


class MseHttBoundarySectionPartialMesh(Frozen):
    r"""A bunch of element faces of the great mesh."""

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

        self._num_global_faces = COMM.allreduce(len(self), op=MPI.SUM)
        self._visualize = None
        self._mn = None
        self._find_cache_ = {}  # cache all!
        self._freeze()

    def info(self, additional_info=''):
        r"""info self."""
        print(
            f"{additional_info}"
            f"msehtt-boundary-section {self._tpm.abstract._sym_repr}: "
            f"{self._num_global_faces} faces"
        )

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._tpm._abstract._sym_repr + super_repr

    def __len__(self):
        r"""how many local element faces?"""
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

        if self._mn == ():
            assert self._num_global_faces == 0, f"must be empty"
            raise Exception(f"num_global_faces={self._num_global_faces}; boundary section is empty!")
        else:
            return self._mn

    def find_dofs(self, f, local=True):
        r"""Find dofs of form `f` on this boundary section.
        """
        if f._is_base():
            key = f.__repr__()
        else:
            key = f._base.__repr__()
        key += str(local)

        if key in self._find_cache_:
            return self._find_cache_[key]
        else:
            pass

        tgm = f.tgm
        assert tgm is self._tgm, f"the great mesh does not match."
        elements = tgm.elements
        degree = f.degree
        space_indicator = f.space.str_indicator
        local_wise_dofs: dict = {}
        for element_index__face_id in self:
            element_index, face_id = element_index__face_id
            if element_index not in local_wise_dofs:
                local_wise_dofs[element_index] = list()
            else:
                pass
            element = elements[element_index]
            local_dofs = element.find_local_dofs_on_face(space_indicator, degree, face_id, component_wise=False)
            local_wise_dofs[element_index].extend(local_dofs)

        for i in local_wise_dofs:
            local_wise_dofs[i] = list(set(local_wise_dofs[i]))
            local_wise_dofs[i].sort()

        if local:
            # if local is True, we return a-local-rank-wise element-wise output, for example:
            # local_wise_dofs =
            # {
            #    14: [5, 11, 17, 23, 29]
            #    ....
            # }
            # this means 14 is a rank element, and the local numbering [5, 11, 17, 23, 29] of form f in great element
            # 14 is on the boundary section.
            #
            # Thus, element #14 will not appear in the output of any other ranks.
            self._find_cache_[key] = local_wise_dofs
            return local_wise_dofs

        else:
            # While if local is False, then we collect the global numbering of all found dofs, gather them and bcast
            # to all ranks. This means we will have the same output (of all found global dofs) for all ranks.
            gm = f.cochain.gathering_matrix
            global_wise_dofs = set()
            for i in local_wise_dofs:
                global_wise_dofs.update(gm[i][local_wise_dofs[i]])
            global_wise_dofs = COMM.gather(global_wise_dofs, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                DOFS = set()
                for dofs in global_wise_dofs:
                    DOFS.update(dofs)
                DOFS = list(DOFS)
                DOFS.sort()
            else:
                DOFS = None

            self._find_cache_[key] = COMM.bcast(DOFS, root=MASTER_RANK)
            return self._find_cache_[key]
