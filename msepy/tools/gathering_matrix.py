# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.tools.miscellaneous.ndarray_cache import ndarray_key_comparer, add_to_ndarray_cache


_cgm_cache = {
    'signatures': '',
    'cgm': None,
}

_msepy_rgm_find_cache = {}


class RegularGatheringMatrix(Frozen):
    """2-d-array: 0-axis -> #elements, 1-axis: local-numbering, values -> global-numbering."""

    def __init__(self, gms, redundant_dof_setting=None, chaining_method=1):
        """"""
        self._num_dofs = None
        if isinstance(gms, np.ndarray):
            self._composite = 1
            assert np.ndim(gms) == 2
            self._gm = gms
            self._gms = (self,)
            self._redundant_dof_setting = self._check_redundant_dof_setting(
                redundant_dof_setting
            )
            _find_cache_signature = self.__repr__()

        elif isinstance(gms, (list, tuple)):
            # we merge some gathering matrices into one
            assert len(gms) > 0, f"cannot chain empty gathering matrices."
            assert redundant_dof_setting is None, \
                f"Can only set redundant_dof info from single gm."

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

                if chaining_method == 0:  # chaining method #0 ----
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

                elif chaining_method == 1:  # most naive method ----------
                    num_dofs = list()
                    total_local_num_dofs = 0
                    for gm in gms:
                        num_dofs.append(gm.num_dofs)
                        total_local_num_dofs += gm.num_local_dofs

                    new_numbering_array = list()
                    for i, gm in enumerate(gms):
                        add2 = sum(num_dofs[:i])
                        new_numbering_array.append(
                            gm._gm + add2
                        )
                    cgm = np.hstack(new_numbering_array)

                else:  # other chaining methods to be implemented.
                    raise NotImplementedError(f"GM chaining method {chaining_method} is not implemented.")

                _cgm_cache['signatures'] = signatures
                _cgm_cache['cgm'] = cgm

            _find_cache_signature = signatures

            self._composite = len(gms)
            self._gm = cgm
            self._gms = gms
            self._redundant_dof_setting = self._parse_redundant_dof_setting(gms)

        else:
            raise NotImplementedError()
        if _find_cache_signature in _msepy_rgm_find_cache:
            pass
        else:
            _msepy_rgm_find_cache[_find_cache_signature] = {}
        self._find_cache = _msepy_rgm_find_cache[_find_cache_signature]
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

    def _check_redundant_dof_setting(self, rds):
        """For example,
            rds = {
                134: ([5], [17]),
                8742: ([254, 967], [0, 667])
                ...
            }

        This means for the global dof #134, it only be the 17th local dof
        of element #5. So, if 134 appears in gm at other places, it means
        nothing. The `_find_elements_and_local_indices_of_dofs` will only
        return the place [(5), (17)].

        For global dof #8742, it is valid at two places: 1) the 0th local
        place at element #254 and 2) the 667th local place at element #967.

        """
        if rds is None:
            rds = {}
        else:
            pass
        assert isinstance(rds, dict), f"Put redundant_dof_setting in dict pls."

        for dof in rds:
            assert isinstance(dof, int) and 0 <= dof < self.num_dofs, \
                f"dof={dof} ({dof.__class__}) is wrong."
            elements, local_indices = rds[dof]
            assert len(elements) == len(local_indices) and len(elements) > 0, \
                f"elements, local indices setting for #{dof} dof wrong."
            for element, index in zip(elements, local_indices):
                assert isinstance(element, int) and 0 <= element < self.num_elements, \
                    f"element={element} out of range."
                assert isinstance(index, int) and 0 <= index, f"index={index} wrong."

        redundant_dof_setting = rds

        return redundant_dof_setting

    def _parse_redundant_dof_setting(self, gms):
        """"""
        total_rds = {}
        local_numbering_starting = 0
        for i, gm in enumerate(gms):
            rds = gm._redundant_dof_setting

            for g_dof in rds:
                elements, local_indices = rds[g_dof]
                element, index = elements[0], local_indices[0]
                total_global_numbering = self[element][index + local_numbering_starting]

                # total_rds[total_global_numbering]
                total_indices = list()
                for index in local_indices:
                    total_indices.append(index + local_numbering_starting)

                total_rds[total_global_numbering] = (elements, total_indices)

            local_numbering_starting += gm.num_local_dofs
        return total_rds

    @property
    def shape(self):
        """the shape of the 2d array gm.

        Raise Error for irregular gathering matrices.
        """
        return self._gm.shape

    @property
    def num_dofs(self):
        """How many dofs in total this gathering matrix represents."""
        if self._num_dofs is None:
            self._num_dofs = int(np.max(self._gm) + 1)
        return self._num_dofs

    @property
    def num_local_dofs(self):
        """How many dofs in each row (each element) of this gathering matrix."""
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
            return rf"<Msepy Gathering Matrix {self.shape} " + super_repr
        else:
            repr_list = list()
            for gm in self._gms:
                repr_list.append(gm.__repr__())
            return 'CHAIN-GM: ' + '-o-'.join(repr_list)

    def __eq__(self, other):
        """"""
        if other.__class__ is not self.__class__:
            return False
        else:
            sgm = self._gm
            ogm = other._gm
            return np.all(sgm == ogm)

    def _find_global_numbering(self, elements, local_dofs):
        """find the global numbering at positions indicated by `elements` and `local_dofs`

        So for element #i (elements[i]), we find the global numbering at locations: local_dofs[i].

        The results of global numbering of dofs will be sorted; thus they are not corresponding to
        the sequence of the inputs.

        """
        global_numbering = set()
        assert len(elements) == len(local_dofs), \
            f"positions are not correct."
        for j, ele in enumerate(elements):
            local_dofs_ele = local_dofs[j]
            global_numbering.update(
                self._gm[ele, local_dofs_ele]
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
                dof: [elements, rows],
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

        if isinstance(dofs, np.ndarray):
            pass
        else:
            dofs = np.array(dofs)

        assert np.ndim(dofs) == 1, f"put all dofs in a 1d data array."

        cached, data = ndarray_key_comparer(
            self._find_cache,
            [dofs]
        )

        if cached:
            return data
        else:
            elements_local_indices = dict()
            for i in dofs:
                assert (0 <= i < self.num_dofs) and (i % 1 == 0), f"dof = {i} is not in range."

                if i in self._redundant_dof_setting:
                    elements_local_indices[i] = self._redundant_dof_setting[i]
                else:
                    elements_local_indices[i] = np.where(self._gm == i)

            add_to_ndarray_cache(
                self._find_cache,
                [dofs],
                elements_local_indices
            )

            return elements_local_indices

    def assemble(self, _2d_array, mode='replace'):
        """Assemble a 2d-array into a 1d array using self._gm."""
        assert _2d_array.shape == self.shape
        _1d_array = np.zeros(self.num_dofs)

        if mode == 'replace':
            for e in range(len(self)):
                element_e_numbering = self._gm[e]
                _1d_array[element_e_numbering] = _2d_array[e]

        else:
            raise NotImplementedError()

        return _1d_array
