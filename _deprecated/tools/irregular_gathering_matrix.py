# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen

_global_ir_cgm_cache = {
    'signatures': '',
    'cgm': dict(),
    'checked': False
}


class IrregularGatheringMatrix(Frozen):
    """"""

    def __init__(self, gms):
        """"""

        if isinstance(gms, dict):
            self._composite = 1
            self._gm = gms
            self._gms = (self,)
        elif isinstance(gms, (list, tuple)):
            # we compose some gathering matrices into one

            num_fcs = None

            signatures = list()

            for gm in gms:
                assert gm.__class__ is IrregularGatheringMatrix, f"can only chain msepy gm."
                signatures.append(gm.__repr__())

                if num_fcs is None:
                    num_fcs = len(gm)
                else:
                    assert num_fcs == len(gm)

            signatures = '-o-'.join(signatures)

            if signatures == _global_ir_cgm_cache['signatures']:
                # the chained gm is same to the previous one, return it from cache.
                cgm = _global_ir_cgm_cache['cgm']

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

                if _global_ir_cgm_cache['checked']:
                    pass
                else:
                    self._check_chain_gm(cgm, gms)
                    _global_ir_cgm_cache['checked'] = True

                _global_ir_cgm_cache['signatures'] = signatures
                _global_ir_cgm_cache['cgm'] = cgm

            self._composite = len(gms)
            self._gm = cgm
            self._gms = gms

        else:
            raise NotImplementedError()

        self._num_dofs = None
        self.___local_shape___ = None
        self._num_local_dofs = {}
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

    def __getitem__(self, i):
        """Return the global_numbering for dofs in element #i."""
        return self._gm[i]

    def __len__(self):
        """How many fundamental cells this gathering_matrix is representing?

        Same to `num_elements`.
        """
        return len(self._gm)

    def __iter__(self):
        """iteration over all local cells."""
        for i in self._gm:
            yield i

    @property
    def _local_shape(self):
        if self.___local_shape___ is None:

            types_pool = []
            shapes = list()

            for index in self:

                ic = index.__class__
                if ic in types_pool:
                    pass
                else:
                    shapes.append(
                        len(self[index])
                    )
                    types_pool.append(ic)

            self.___local_shape___ = shapes

        return self.___local_shape___

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

    def num_local_dofs(self, i):
        """How many local dofs in cell #i."""
        if i in self._num_local_dofs:
            return self._num_local_dofs[i]
        else:
            _ = len(self[i])
            self._num_local_dofs[i] = _
            return _

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
                    fc_indices = [i for _ in local_indices]
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
