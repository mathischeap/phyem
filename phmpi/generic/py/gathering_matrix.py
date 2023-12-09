# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM, SIZE, MPI
from scipy.sparse import lil_matrix

_global_MPI_PY_cgm_cache = {
    'signatures': '',
    'cgm': dict(),
    'checked': False,
}


class MPI_PyGM(Frozen):
    """The mpi (distributed) version of the python gathering matrix."""

    def __init__(self, *gms):
        """"""
        if len(gms) == 1:
            gm = gms[0]

            if gm.__class__ is MPI_PyGM:
                gm = gm._gm
                gms = gms
            else:
                assert isinstance(gm, dict), f"pls put gm in dict."
                gm = gm
                gms = [self, ]

        else:
            for i, gm_i in enumerate(gms):
                assert gm_i.__class__ is MPI_PyGM, f"gms[{i}] is not a {MPI_PyGM}."
            num_elements = None
            signatures = list()
            for gm in gms:
                signatures.append(gm.__repr__())
                if num_elements is None:
                    num_elements = len(gm)
                else:
                    assert num_elements == len(gm), f'gms element amount dis-match.'

            signatures = '-o-'.join(signatures)
            if signatures == _global_MPI_PY_cgm_cache['signatures']:
                # the chained gm is same to the previous one, return it from cache.
                cgm = _global_MPI_PY_cgm_cache['cgm']
            else:
                # ---------------------------------------------------------------------------------
                num_local_dofs = dict()
                for i in gms[0]:
                    _ = 0
                    for gm in gms:
                        _ += gm.num_local_dofs(i)
                    num_local_dofs[i] = _

                for gm in gms:   # make sure num_dofs are made. It needs global communication!
                    _ = gm.num_dofs

                cache = None   # make sure we raise Error if something unexpected happens.

                for rank in range(SIZE):
                    if RANK == rank:
                        if RANK == 0:
                            cache = list()
                            for gm in gms:
                                cache.append(-np.ones(gm.num_dofs, dtype=int))
                            current_number = 0
                            filters = [0 for _ in range(len(gms))]
                        else:
                            cache = COMM.recv(source=RANK-1, tag=RANK+SIZE)
                            current_number = COMM.recv(source=RANK-1, tag=RANK+SIZE+1)
                            filters = COMM.recv(source=RANK-1, tag=RANK+SIZE+2)

                        for i in gms[0]:
                            for j, gm in enumerate(gms):
                                fv = gm[i]
                                Filter = filters[j]
                                fv = fv[fv >= Filter]  # these dofs should be re-numbered
                                amount_re_numbered = len(fv)
                                filters[j] += amount_re_numbered
                                cache[j][fv] = np.arange(current_number, current_number+amount_re_numbered)
                                current_number += amount_re_numbered

                        if RANK < SIZE - 1:
                            COMM.send(cache, dest=RANK+1, tag=RANK+1+SIZE)
                            COMM.send(current_number, dest=RANK+1, tag=RANK+1+SIZE+1)
                            COMM.send(filters, dest=RANK+1, tag=RANK+1+SIZE+2)
                        else:
                            pass
                    else:
                        pass

                cgm = dict()
                for i in gms[0]:
                    cgm[i] = - np.ones(num_local_dofs[i], dtype=int)

                for i in gms[0]:
                    local_indices = 0
                    for j, gm in enumerate(gms):
                        global_dofs_j = gm[i]
                        renumbering = cache[j][global_dofs_j]
                        local_dofs_j = len(global_dofs_j)
                        cgm[i][local_indices:local_indices+local_dofs_j] = renumbering
                        local_indices += local_dofs_j

                for i in cgm:
                    assert -1 not in cgm[i], f"all dofs are numbered."

                if _global_MPI_PY_cgm_cache['checked']:
                    pass
                else:
                    self._check_chain_gm(cgm, gms)
                    _global_MPI_PY_cgm_cache['checked'] = True

                _global_MPI_PY_cgm_cache['signatures'] = signatures
                _global_MPI_PY_cgm_cache['cgm'] = cgm
                # ================================================================================
            gm = cgm
            gms = gms

        # -----------------------------------------------------------------------
        assert isinstance(gm, dict), f"put raw gathering matrix in a dictionary. {gm.__class__}"
        for gmi in gms:
            assert gmi.__class__ is MPI_PyGM, f"must be"
        for i in gm:
            assert isinstance(gm[i], np.ndarray) and np.ndim(gm[i]) == 1, \
                f"numbering of element #{i} must be a 1d ndarray."
        # -----------------------------------------------------------------------

        self._gm = gm
        self._gms = gms
        self._num_dofs = None
        self._num_local_dofs = dict()
        self._freeze()

    @staticmethod
    def _check_chain_gm(cgm, gms):
        """"""
        # -- first check the total dofs
        num_dofs = 0
        for gm in gms:
            num_dofs += gm.num_dofs
        cgm_num_dofs = list()
        for index in cgm:
            cgm_num_dofs.append(
                max(cgm[index])
            )
        cgm_num_dofs = max(cgm_num_dofs) + 1
        cgm_num_dofs = COMM.allreduce(cgm_num_dofs, op=MPI.MAX)
        assert cgm_num_dofs == num_dofs, f'Total num dofs wrong'

        if cgm_num_dofs > 50000:
            return
        else:
            pass

        # ---- carefully check it.
        numbering_pool = None
        for rank in range(SIZE):
            if RANK == rank:
                if RANK == 0:
                    checking_dict = dict()
                    numbering_pool = list()
                else:
                    checking_dict = COMM.recv(source=RANK-1, tag=RANK+SIZE)
                    numbering_pool = COMM.recv(source=RANK-1, tag=RANK+SIZE+1)

                for index in cgm:
                    j = 0
                    cgm_numbering = cgm[index]
                    for k, gm in enumerate(gms):
                        gm_numbering = gm[index]
                        for m in gm_numbering:
                            key = (k, m)
                            c_numbering = cgm_numbering[j]
                            if key in checking_dict:
                                assert checking_dict[key] == c_numbering, f"something is wrong."
                            else:
                                checking_dict[key] = c_numbering
                            numbering_pool.append(c_numbering)
                            j += 1

                if RANK < SIZE - 1:
                    COMM.send(checking_dict, dest=RANK+1, tag=RANK+1+SIZE)
                    COMM.send(numbering_pool, dest=RANK+1, tag=RANK+1+SIZE+1)
                else:
                    pass
            else:
                pass

        if RANK == SIZE-1:  # the last rank
            numbering_pool = list(set(numbering_pool))
            numbering_pool.sort()
            assert numbering_pool == [_ for _ in range(cgm_num_dofs)], f"numbering is not continuous!"

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<MPI-PY2-PyGM of {self.num_elements} elements" + super_repr

    @property
    def num_dofs(self):
        """the amount of total dofs across all cores."""
        if self._num_dofs is None:
            local_max = list()
            for index in self:
                local_max.append(max(self[index]))
            local_max = max(local_max)
            local_max = COMM.gather(local_max, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                local_max = max(local_max) + 1
            else:
                local_max = None
            self._num_dofs = COMM.bcast(local_max, root=MASTER_RANK)
        return self._num_dofs

    def __iter__(self):
        """Iteration over all local element indices."""
        for index in self._gm:
            yield index

    def __contains__(self, index):
        """If element #index is a local element?"""
        return index in self._gm

    def __getitem__(self, index):
        """The global numbering of dos in element #index."""
        return self._gm[index]

    def __len__(self):
        """How many local elements I am representing?"""
        return len(self._gm)

    def num_local_dofs(self, index):
        """Num of local dofs in element #index."""
        if index not in self._num_local_dofs:
            self._num_local_dofs[index] = len(self[index])
        return self._num_local_dofs[index]

    @property
    def num_elements(self):
        """How many local elements?"""
        return len(self)

    def __eq__(self, other):
        """check if two gathering matrices are equal to each other."""
        if self is other:
            local_tof = True
        else:
            if other.__class__ is not self.__class__:
                local_tof = False
            elif other._gm is self._gm:
                local_tof = True
            else:
                if len(self) != len(other):
                    local_tof = False
                else:
                    local_tof = True
                    for i in self:
                        if i not in other:
                            local_tof = False
                            break
                        else:
                            if np.all(self[i] == other[i]):
                                pass
                            else:
                                local_tof = False
                                break

        return COMM.allreduce(local_tof, op=MPI.LAND)

    def assemble(self, data_dict, mode='replace', globalize=True):
        """Assemble a 2d-array into a 1d array using self._gm.

        Parameters
        ----------
        data_dict :
            The data to be assembled
        mode :
            {'replace',}
            if `mode` == 'replace`, then when multiple data appear at the same place, we use one of the data
            (instead of sum them up).

        globalize : bool
            If `globalize` is True, we return the same global 1d array in all cores.

        """
        assert isinstance(data_dict, dict) and len(data_dict) == len(self)
        _ = self.num_dofs

        if mode == 'replace':

            if globalize:

                dict_numbering_data = dict()
                for element_index in self:
                    element_numbering = self._gm[element_index]
                    dict_numbering_data[element_index] = [element_numbering, data_dict[element_index]]

                dict_numbering_data = COMM.gather(dict_numbering_data, root=MASTER_RANK)

                if RANK == MASTER_RANK:
                    LIL = lil_matrix((1, self.num_dofs))
                    for local_numbering_data in dict_numbering_data:
                        for index in local_numbering_data:
                            element_numbering, data = local_numbering_data[index]
                            LIL[0, element_numbering] = data
                    _1d_array = LIL.toarray().ravel('C')
                else:
                    _1d_array = np.zeros(self.num_dofs, dtype=float)

                COMM.Bcast([_1d_array, MPI.FLOAT], root=MASTER_RANK)

                return _1d_array

            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    def _find_global_numbering(self, element_indices, local_dofs):
        """find the global numbering at positions indicated by `element_indices` and `local_dofs`

        So for element #i (element_indices[i]), we find the global numbering of dofs locally at local_dofs[i]

        The global numbering then are sorted. So the sequence are not consistent with the inputs.

        For example:
            element_indices = [
                0,
                3,
                '5=2-0',
                ...,
            ]

            local_dofs = [
                [0, 1, 2],           # local dofs of element #0
                [1, 4, 7, 10],       # local dofs of element #3
                [2, 3, 4, 5],        # local dofs of element #'5=2-0'
                ...,
            ]

        So, we will try to found global numbering of local dof#0, #1, #2 in element #0 and dofs local #1, #4 ...

        The output will be gathered together and bcast to all ranks.

        """
        global_numbering = set()
        assert len(element_indices) == len(local_dofs), f"positions are not correct."
        for j, element_index in enumerate(element_indices):
            if element_index in self:
                local_dofs_ele = local_dofs[j]
                global_numbering.update(
                    self._gm[element_index][local_dofs_ele]
                )
            else:
                pass
        global_numbering = COMM.gather(global_numbering, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            GLOBAL_NUMERING = set()
            for _ in global_numbering:
                GLOBAL_NUMERING.update(_)
            GLOBAL_NUMERING = list(GLOBAL_NUMERING)
            GLOBAL_NUMERING.sort()
        else:
            GLOBAL_NUMERING = None
        return COMM.bcast(GLOBAL_NUMERING, root=MASTER_RANK)

    def _find_elements_and_local_indices_of_dofs(self, dofs):
        """Return all elements and local indices in all cores regardless whether the elements are local.

        This means we need send the same inputs to all ranks.

        So, return ths same output in all cores.

        Parameters
        ----------
        dofs

        Returns
        -------

        """
        if isinstance(dofs, int):
            if dofs < 0:
                dofs = self.num_dofs + dofs
            else:
                pass
            dofs = [dofs, ]

        else:
            pass

        elements_local_indices = dict()

        for d in dofs:
            assert (0 <= d < self.num_dofs) and (d % 1 == 0), f"dof = {d} is not in range."

        for d in dofs:
            elements_local_indices[d] = ([], [])

        for i in self:
            local_gm = self[i]
            for d in dofs:
                if d in local_gm:
                    local_indices = list(np.where(local_gm == d)[0])
                    elements = [i for _ in local_indices]
                    elements_local_indices[d][0].extend(elements)
                    elements_local_indices[d][1].extend(local_indices)
                else:
                    pass

        elements_local_indices = COMM.gather(elements_local_indices, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            complete = dict()

            for d in dofs:
                elements = list()
                local_indices = list()

                for rank_elements_indices in elements_local_indices:

                    rank_elements, rank_indices = rank_elements_indices[d]

                    elements.extend(rank_elements)
                    local_indices.extend(rank_indices)

                complete[d] = (elements, local_indices)

        else:
            complete = None

        return COMM.bcast(complete, root=MASTER_RANK)
