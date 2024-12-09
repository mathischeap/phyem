# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen

from src.config import RANK, MASTER_RANK, COMM, SIZE

from msehtt.static.mesh.partial.elements.visualize.main import MseHttElementsPartialMeshVisualize


class MseHtt_NCF_Elements_PartialMesh(Frozen):
    """The partial mesh that contains only great elements (all or partial)."""

    def __init__(self, tpm, tgm, local_element_range):
        """"""
        self._tpm = tpm
        self._tgm = tgm
        for i in local_element_range:
            assert i in tgm.elements, f"element #{i} is not a valid local great mesh element."
        self._range = local_element_range
        self._merge_element_indices()
        self._mn = None
        self._visualize = None
        self._freeze()

    def info(self):
        """info self."""
        print(f"msehtt-ncf-partial-elements > {self._tpm.abstract._sym_repr}: "
              f"{self._num_global_elements} elements > distributed in {SIZE} ranks.")

    @property
    def ___is_msehtt_ncf_partial_elements_mesh___(self):
        r"""Just a signature"""
        return True

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._tpm._abstract._sym_repr + super_repr

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
        r""""""
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

    @property
    def visualize(self):
        """To visualize this partial mesh of elements."""
        if self._visualize is None:
            self._visualize = MseHttElementsPartialMeshVisualize(self)
        return self._visualize
