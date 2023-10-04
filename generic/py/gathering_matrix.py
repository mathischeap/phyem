# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen

_global_cgm_cache = {
    'signatures': '',
    'cgm': dict(),
    'checked': False,
}


class PyGM(Frozen):
    """"""

    def __init__(self, *gms):
        """"""
        if len(gms) == 1:
            gm = gms[0]

            if gm.__class__ is PyGM:
                gm = gm._gm
                gms = gms
            else:
                assert isinstance(gm, dict), f"put gm in dict."
                gm = gm
                gms = [self, ]

        else:
            for i, gm_i in enumerate(gms):
                assert gm_i.__class__ is PyGM, f"gms[{i}] is not a {PyGM}."

            num_elements = None
            signatures = list()
            for gm in gms:
                signatures.append(gm.__repr__())
                if num_elements is None:
                    num_elements = len(gm)
                else:
                    assert num_elements == len(gm)

            signatures = '-o-'.join(signatures)
            if signatures == _global_cgm_cache['signatures']:
                # the chained gm is same to the previous one, return it from cache.
                cgm = _global_cgm_cache['cgm']
            else:
                num_local_dofs = dict()
                for i in gms[0]:
                    _ = 0
                    for gm in gms:
                        _ += gm.num_local_dofs(i)
                    num_local_dofs[i] = _

                cache = list()
                for gm in gms:
                    cache.append(np.ones(gm.num_dofs, dtype=int))

                current_number = 0

                filters = [0 for _ in range(len(gms))]

                for i in gms[0]:
                    for j, gm in enumerate(gms):
                        fv = gm[i]
                        Filter = filters[j]
                        fv = fv[fv >= Filter]  # these dofs should be re-numbered
                        amount_re_numbered = len(fv)
                        filters[j] += amount_re_numbered
                        cache[j][fv] = np.arange(current_number, current_number+amount_re_numbered)
                        current_number += amount_re_numbered

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

                if _global_cgm_cache['checked']:
                    pass
                else:
                    self._check_chain_gm(cgm, gms)
                    _global_cgm_cache['checked'] = True
                _global_cgm_cache['signatures'] = signatures
                _global_cgm_cache['cgm'] = cgm

            gm = cgm
            gms = gms
        # ---------------------------------------------------------------------------------------
        assert isinstance(gm, dict), f"put raw gathering matrix in a dictionary. {gm.__class__}"
        for gmi in gms:
            assert gmi.__class__ is PyGM, f"must be"
        for i in gm:
            assert isinstance(gm[i], np.ndarray) and np.ndim(gm[i]) == 1, \
                f"numbering of element #{i} must be a 1d ndarray."
        # ---------------------------------------------------------------------------------------
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

        if cgm_num_dofs > 10000:
            return

        assert cgm_num_dofs == num_dofs, f'Total num dofs wrong'
        # ---- carefully check it.
        checking_dict = dict()
        numbering_pool = list()
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

        numbering_pool = list(set(numbering_pool))
        numbering_pool.sort()
        assert numbering_pool == [_ for _ in range(cgm_num_dofs)]

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<PyGM of {self.num_elements} elements" + super_repr

    def __getitem__(self, index):
        """"""
        return self._gm[index]

    def __len__(self):
        """How many elements in total?"""
        return len(self._gm)

    @property
    def num_elements(self):
        """How many elements in total?"""
        return len(self)

    def __contains__(self, index):
        return index in self._gm

    def __iter__(self):
        for index in self._gm:
            yield index

    def num_local_dofs(self, index):
        """how many local dofs in element #index."""
        if index not in self._num_local_dofs:
            self._num_local_dofs[index] = len(self[index])
        return self._num_local_dofs[index]

    @property
    def num_dofs(self):
        """total num dofs."""
        if self._num_dofs is None:
            max_list = list()
            for index in self:
                max_list.append(max(self[index]))
            self._num_dofs = max(max_list) + 1
        return self._num_dofs

    def __eq__(self, other):
        """"""
        if self is other:
            return True
        else:
            if other.__class__ is not self.__class__:
                return False
            else:
                if len(self) != len(other):
                    return False
                else:
                    for i in self:
                        if i not in other:
                            return False
                        else:
                            if np.all(self[i] == other[i]):
                                pass
                            else:
                                return False
                    return True

    def _find_global_numbering(self, fundamental_cells, local_dofs):
        """find the global numbering at positions indicated by `fundamental_cells` and `local_dofs`

        So for fundamental_cell #i (fundamental_cells[i]), we find the global numbering at locations: local_dofs[i]
        """
        global_numbering = set()
        assert len(fundamental_cells) == len(local_dofs), f"positions are not correct."
        for j, fc_indices in enumerate(fundamental_cells):
            local_dofs_ele = local_dofs[j]
            global_numbering.update(
                self._gm[fc_indices][local_dofs_ele]
            )
        global_numbering = list(global_numbering)
        global_numbering.sort()
        return global_numbering

    def _find_elements_and_local_indices_of_dofs(self, dofs):
        """

        Parameters
        ----------
        dofs

        Returns
        -------
        A dict:
            {
                dof: [fc_indices, local_indices],
            }
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

        return elements_local_indices
