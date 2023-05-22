# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""
import numpy as np
from tools.frozen import Frozen


_cgm_cache = {
    'signatures': '',
    'cgm': None,
}


class RegularGatheringMatrix(Frozen):
    """2-d-array: 0-axis -> #elements, 1-axis: local-numbering, values -> global-numbering."""

    def __init__(self, gms):
        """"""
        if isinstance(gms, np.ndarray):
            self._composite = 1
            assert np.ndim(gms) == 2
            self._gm = gms
            self._gms = (self,)
        elif isinstance(gms, (list, tuple)):
            # we compose some gathering matrices into one

            num_elements = None
            num_local_dofs = 0

            signatures = list()
            for gm in gms:
                assert gm.__class__ is RegularGatheringMatrix, f"can only chain msepy gm."

                signatures.append(gm.__repr__())

                if num_elements is None:
                    num_elements = gm.num_elements
                else:
                    assert num_elements == gm.num_elements
                num_local_dofs += gm.num_local_dofs

            signatures = '-o-'.join(signatures)

            if signatures == _cgm_cache['signatures']:
                cgm = _cgm_cache['cgm']

            else:

                cache = list()
                for gm in gms:
                    cache.append(np.ones(gm.num_dofs, dtype=int))

                current_number = 0
                filters = [0 for _ in range(len(gms))]
                for i in range(num_elements):
                    for j, gm in enumerate(gms):
                        fv = gm[i]
                        Filter = filters[j]
                        fv = fv[fv >= Filter]  # these dofs should be re-numbered
                        amount_re_numbered = len(fv)
                        filters[j] += amount_re_numbered
                        cache[j][fv] = np.arange(current_number, current_number+amount_re_numbered)
                        current_number += amount_re_numbered

                cgm = - np.ones((num_elements, num_local_dofs), dtype=int)
                for i in range(num_elements):
                    local_indices = 0
                    for j, gm in enumerate(gms):
                        global_dofs_j = gm[i]
                        renumbering = cache[j][global_dofs_j]
                        local_dofs_j = gm.num_local_dofs
                        cgm[i][local_indices:local_indices+local_dofs_j] = renumbering
                        local_indices += local_dofs_j
                _cgm_cache['signatures'] = signatures
                _cgm_cache['cgm'] = cgm

            self._composite = len(gms)
            self._gm = cgm
            self._gms = gms
        else:
            raise NotImplementedError()
        self._num_dofs = None
        self._freeze()

    def __getitem__(self, i):
        """Return the global_numbering for dofs in element #i."""
        return self._gm[i]

    def __len__(self):
        """How many elements this gathering_matrix is representing?

        Same to `num_elements`.
        """
        return self.num_elements

    def __iter__(self):
        """iteration over all local elements."""
        for i in range(self.num_elements):
            yield i

    @property
    def shape(self):
        """the shape of the 2d array gm.

        Raise Error for irregular gathering matrices.
        """
        return self._gm.shape

    @property
    def num_dofs(self):
        """"""
        if self._num_dofs is None:
            self._num_dofs = int(np.max(self._gm) + 1)
        return self._num_dofs

    @property
    def num_local_dofs(self):
        """"""
        return self.shape[1]

    @property
    def num_elements(self):
        """How many elements this gathering_matrix is representing?

        Same to `len`.
        """
        return self.shape[0]

    def __repr__(self):
        """repr"""
        if self._composite == 1:
            super_repr = super().__repr__().split(' object ')[1]
            return rf"<base msepy gathering matrix " + super_repr
        else:
            repr_list = list()
            for gm in self._gms:
                repr_list.append(gm.__repr__())
            return '-o-'.join(repr_list)

    def __eq__(self, other):
        """"""
        if other.__class__ is not self.__class__:
            return False
        else:
            sgm = self._gm
            ogm = other._gm
            return np.all(sgm == ogm)

    def _find_elements_and_local_indices_of_dofs(self, dofs):
        """"""
        if isinstance(dofs, int):
            if dofs < 0:
                dofs = self.num_dofs + dofs
            else:
                pass

            dofs = [dofs, ]
        else:
            pass

        elements_local_indices = dict()

        for i in dofs:
            assert (0 <= i < self.num_dofs) and (i % 1 == 0), f"dof = {i} is not in range."

            elements_local_indices[i] = np.where(self._gm == i)

        return elements_local_indices
