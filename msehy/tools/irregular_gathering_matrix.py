# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class IrregularGatheringMatrix(Frozen):
    """"""

    def __init__(self, gms):
        """"""
        if isinstance(gms, dict):
            self._composite = 1
            self._gm = gms
            self._gms = (self,)
        else:
            raise NotImplementedError()

        self._num_dofs = None
        self._freeze()

    def __getitem__(self, i):
        """Return the global_numbering for dofs in element #i."""
        return self._gm[i]

    def __len__(self):
        """How many fundamental cells this gathering_matrix is representing?

        Same to `num_elements`.
        """
        return len(self._gm)

    def __iter__(self):
        """iteration over all local elements."""
        for i in self._gm:
            yield i

    @property
    def num_dofs(self):
        """How many dofs in total this gathering matrix represents."""
        if self._num_dofs is None:
            local_max = list()
            for i in self:
                local_max.append(
                    max(self[i])
                )
            self._num_dofs = int(max(local_max)) + 1
        return self._num_dofs

    def __repr__(self):
        """repr"""
        if self._composite == 1:
            super_repr = super().__repr__().split(' object ')[1]
            return rf"<iGM: num fc{len(self)}, num dofs {self.num_dofs} " + super_repr
        else:
            repr_list = list()
            for gm in self._gms:
                repr_list.append(gm.__repr__())
            return '<CHAIN-iGM=' + '-o-'.join(repr_list) + ">"

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

    def _find_fundamental_cells_and_local_indices_of_dofs(self, dofs):
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

        fc_indices_local_indices = dict()

        for d in dofs:
            assert (0 <= d < self.num_dofs) and (d % 1 == 0), f"dof = {d} is not in range."

        for d in dofs:
            fc_indices_local_indices[d] = ([], [])

        for i in self:
            local_gm = self[i]
            for d in dofs:
                if d in local_gm:

                    local_indices = list(np.where(local_gm == d)[0])
                    fc_indices = np.ones_like(local_indices)
                    fc_indices_local_indices[d][0].extend(fc_indices)
                    fc_indices_local_indices[d][1].extend(local_indices)

                else:
                    pass

        return fc_indices_local_indices

    def assemble(self, data_dict, mode='replace'):
        """Assemble a 2d-array into a 1d array using self._gm."""
        _1d_array = np.zeros(self.num_dofs)
        if mode == 'replace':
            for i in self:
                element_e_numbering = self[i]
                _1d_array[element_e_numbering] = data_dict[i]

        else:
            raise NotImplementedError()

        return _1d_array
