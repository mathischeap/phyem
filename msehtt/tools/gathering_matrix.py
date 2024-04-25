# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from src.config import COMM, RANK, MASTER_RANK, SIZE, MPI

___cache_msehtt_gm_chaining___ = {}
___cache_msehtt_gm_find___ = {}


def ___clean_cache_msehtt_gm___():
    keys = list(___cache_msehtt_gm_chaining___.keys())
    for key in keys:
        del ___cache_msehtt_gm_chaining___[key]
    keys = list(___cache_msehtt_gm_find___.keys())
    for key in keys:
        del ___cache_msehtt_gm_find___[key]


def ___msehtt_gm_chaining_method_0___(gms):
    """"""
    if RANK == MASTER_RANK:
        num_global_dofs = list()
        ALL_GM = dict()
    else:
        pass

    num_variables = len(gms)

    for i, gm in enumerate(gms):
        all_gm = COMM.gather(gm._gm, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            # noinspection PyUnboundLocalVariable
            num_global_dofs.append(gm.num_global_dofs)
            # noinspection PyUnboundLocalVariable
            ALL_GM[i] = dict()
            for _ in all_gm:
                ALL_GM[i].update(_)
        else:
            pass

    rank_elements = COMM.gather(gms[0]._gm.keys(), root=MASTER_RANK)

    if RANK == MASTER_RANK:
        total_global_dofs = sum(num_global_dofs)
        # the total amount (across all ranks) of dofs in the chained gm

        pool_dict = dict()
        current = 0
        gm_to_be_distributed = [{} for _ in range(SIZE)]
        for k, elements in enumerate(rank_elements):
            the_gm = gm_to_be_distributed[k]
            for e in elements:
                the_gm[e] = list()
                for i in range(num_variables):
                    numbering = ALL_GM[i][e]
                    for num in numbering:
                        key = (i, num)
                        if key in pool_dict:
                            pass
                        else:
                            pool_dict[key] = current
                            current += 1
                        the_gm[e].append(pool_dict[key])
                the_gm[e] = np.array(the_gm[e], dtype=int)
        assert current == total_global_dofs, f"must be like this."
    else:
        gm_to_be_distributed = None
    cgm = COMM.scatter(gm_to_be_distributed, root=MASTER_RANK)
    return cgm


def ___msett_gm_chaining___(gms, method=0):
    """"""
    assert len(gms) >= 2, f"make sense only when chaining more than 1 gm."
    # ---- check element consistence in all ranks --------------------------------------
    element_indices = gms[0]._gm.keys()

    for j, gm in enumerate(gms[1:]):
        assert gm.keys() == element_indices, f"gm[{j+1}] element indices dis-match that of gm[0]."

    if method == 0:
        return ___msehtt_gm_chaining_method_0___(gms)
    else:
        raise NotImplementedError(f"chaining method = {method} is not implemented.")


class MseHttGatheringMatrix(Frozen):
    """"""
    def __init__(self, gms):
        """"""
        self._num_global_dofs = None
        self._num_rank_dofs = None

        if isinstance(gms, dict):
            self._composite = 1
            self._signature = self.__repr__()
            self._gms = (self,)
            self._gm = gms

        else:
            isinstance(gms, (list, tuple)), f"when chaining multiple gms, put them in a list or tuple."
            assert len(gms) > 0, f"Chain empty gathering matrix?"
            self._composite = len(gms)
            signatures = list()
            for i, gm in enumerate(gms):
                assert gm.__class__ is self.__class__, f"gms[i]={gm} is not a {self.__class__}."
                signatures.append(gm._signature)
            signatures = '-o-'.join(signatures)
            self._signature = signatures
            self._gms = gms

            if len(gms) == 1:
                cgm = gms[0]._gm
            else:
                # chaining multiple gathering matrices.
                if self._signature in ___cache_msehtt_gm_chaining___:
                    cgm = ___cache_msehtt_gm_chaining___[self._signature]
                else:
                    cgm = ___msett_gm_chaining___(gms, method=0)
                    ___cache_msehtt_gm_chaining___[self._signature] = cgm
            self._gm = cgm

        if self._signature in ___cache_msehtt_gm_find___:
            pass
        else:
            ___cache_msehtt_gm_find___[self._signature] = {}
        self._find_cache_ = ___cache_msehtt_gm_find___[self._signature]
        self._check_gm_and_gms()
        self._freeze()

    def __repr__(self):
        """Different for all gm, but different gm could have same signature."""
        if self._composite == 1:
            super_repr = super().__repr__().split(' object ')[1]
            return rf"<MseHtt-Gathering-Matrix " + super_repr
        else:
            repr_list = list()
            for gm in self._gms:
                repr_list.append(gm.__repr__())
            super_repr = super().__repr__().split(' object ')[1]
            return r'CHAIN-GM ::: ' + '-o-'.join(repr_list) + ' ::: ' + super_repr

    def _check_gm_and_gms(self):
        """"""
        assert isinstance(self._gm, dict), f"MseTT `_gm` must be a dict."
        assert all([_.__class__ is self.__class__ for _ in self._gms]), \
            f"each of `gms` must be {self.__class__}."

        # ----- we first check all ranks contain different elements -------------------------
        self._num_rank_elements = len(self._gm)

        if self._composite == 1 and self._gms[0] is self:  # only check root gm.
            local_elements = set(self._gm.keys())
            all_local_elements = COMM.gather(local_elements, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                for i in range(SIZE):
                    for j in range(i+1, SIZE):
                        elements0, elements1 = all_local_elements[i], all_local_elements[j]
                        if len(elements0.intersection(elements1)) == 0:
                            pass
                        else:
                            raise Exception(f"same element appear in RANKS {i} and {j}.")

            else:
                pass

        self._num_global_elements = COMM.allreduce(self._num_rank_elements, op=MPI.SUM)

        # ---- check the numbering for each element ------------------------------------------
        max_numbering = -1
        zero_local_numbering = np.array([], dtype=int)
        allowed_dtypes = (
            np.dtypes.Int32DType, np.dtypes.Int64DType
        )
        for i in self:  # go through all local elements
            numbering = self[i]  # the numbering of element #i
            if self._composite == 1 and self._gms[0] is self:  # only check root gm.
                if numbering is None:  # no valid dof in this element.
                    self._gm[i] = zero_local_numbering  # replace None by an empty array.
                else:
                    assert isinstance(numbering, np.ndarray), \
                        f"The numbering of element must be a 1-d array."

                    if len(numbering) == 0:  # no valid dof in this element.
                        pass
                    else:
                        assert np.ndim(numbering) == 1, f"The numbering of element must be a 1-d array."
                        assert np.min(numbering) >= 0, f"numbering must starts with 0."
                        assert numbering.dtype.__class__ in allowed_dtypes, \
                            f"Not allowed dtype."
            else:
                pass

            if len(self[i]) > 0:  # it is not empty.
                max_numbering = max([max_numbering, np.max(numbering)])
            else:
                pass

        max_numbering = COMM.allgather(max_numbering)
        self._num_global_dofs = max(max_numbering) + 1

    @property
    def num_global_dofs(self):
        """total dofs across all ranks. Return the same value for in all ranks."""
        return self._num_global_dofs

    @property
    def num_rank_dofs(self):
        """dofs in this rank."""
        if self._num_rank_dofs is None:
            raise NotImplementedError()  # to be implemented!
        return self._num_rank_dofs

    def num_local_dofs(self, i):
        """the num local dofs in element #`i`."""
        return len(self._gm[i])

    @property
    def num_global_elements(self):
        """The total amount of elements across all ranks."""
        return self._num_global_elements

    @property
    def num_rank_elements(self):
        """The amount of elements in this rank."""
        return self._num_rank_elements

    def __getitem__(self, i):
        """"""
        return self._gm[i]  # the numbering of element #`i`.

    def __iter__(self):
        """go through all rank elements."""
        for i in self._gm:
            yield i

    def __contains__(self, i):
        """check if element #`i` is a local element."""
        return i in self._gm

    def __len__(self):
        """"""
        return self._num_rank_elements

    def __eq__(self, other):
        """Check if two gathering matrices are equal."""
        if other.__class__ is not self.__class__:
            rank_true_or_false = False
        else:
            if self is other:
                rank_true_or_false = True
            else:
                if self._signature == other._signature:
                    rank_true_or_false = True
                else:
                    sgm = self._gm
                    ogm = other._gm
                    if len(sgm) == len(ogm):
                        rank_true_or_false = True
                        for i in sgm:
                            if i not in ogm:
                                rank_true_or_false = False
                                break
                            else:
                                s_numbering = sgm[i]
                                o_numbering = ogm[i]
                                if np.all(s_numbering == o_numbering):
                                    pass
                                else:
                                    rank_true_or_false = False
                                    break
                    else:
                        rank_true_or_false = False

        return COMM.allreduce(rank_true_or_false, op=MPI.LAND)

    def find_global_numbering_of_local_dofs(self, elements, local_indices):
        """Each element in `elements` must be a local. So, no communication
        take place in this method.

        Parameters
        ----------
        elements
        local_indices

        Returns
        -------

        """
        raise NotImplementedError()

    def find_rank_locations_of_global_dofs(self, global_dofs):
        """find all the locations in this rank of global dofs. So this is a local
        process, no communication is needed.

        It returns a dict like:
            {
                98 : ((3, 15), (8, 7)),
                100: (),
                ....,
            }
        This means the global dof #98 in this rank has two locations:
            1) (3, 15) i.e. the 15th local dof of element #3.
            2) (8, 7) i.e. the 7th local dof of element #8.

        And the global dof #100 is not in this local rank.

        """
        if isinstance(global_dofs, (int, float)):
            global_dofs = [global_dofs,]
        else:
            pass

        location_dict = {}
        for gd in global_dofs:
            if gd in self._find_cache_:
                location_dict[gd] = self._find_cache_[gd]
            else:
                assert isinstance(gd, (int, float)), f"dof #{gd} wrong, it must be int."
                if isinstance(gd, float):
                    assert gd % 1 == 0, f"dof #{gd} wrong. It must be int."
                else:
                    pass
                assert gd >= 0, f"dof #{gd} wrong. It must be non-negative int."
                temp = list()
                for e in self:
                    numbering = self[e]
                    if gd in numbering:
                        indices = list(np.where(numbering == gd)[0])
                        for i in indices:
                            temp.append((e, i))
                    else:
                        pass
                location_dict[gd] = tuple(temp)
                self._find_cache_[gd] = location_dict[gd]

        return location_dict

    def assemble(self, data):
        """Assemble a data structure in a _1d array."""
        raise NotImplementedError()
