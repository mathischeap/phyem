# -*- coding: utf-8 -*-
r"""
"""
from typing import Dict
import numpy as np
from tools.frozen import Frozen
from legacy.generic.py.gathering_matrix import PyGM
from legacy.generic.py.vector.globalize.static import Globalize_Static_Vector


class Localize_Static_Vector(Frozen):
    """"""

    def __init__(self, localized_vector, gm):
        """"""
        assert gm.__class__ is PyGM, f'gm is not a {PyGM}'

        # ------------- initialize a empty vector --------------------------------------------
        if isinstance(localized_vector, (int, float)) and localized_vector == 0:
            localized_vector = dict()
            _cache = {}
            for index in gm:
                _num_local_dofs = gm.num_local_dofs(index)
                if _num_local_dofs in _cache:
                    pass
                else:
                    _cache[_num_local_dofs] = np.zeros(_num_local_dofs)
                localized_vector[index] = _cache[_num_local_dofs]
        else:
            pass
        # ===================================================================================

        self._meta_data = None
        self._gm = gm
        self._set_data(localized_vector)
        self._adjust = _Adjust(self)
        self._customize = _Customize(self)
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Localize_Static_Vector over {len(self)} elements{super_repr}>"

    def _set_data(self, localized_vector):
        """"""
        if localized_vector is None:
            self._meta_data = None
            self._dtype = 'None'

        elif isinstance(localized_vector, dict):
            # -------- data check --------------------------------------------------------------
            assert len(localized_vector) == len(self._gm), f"data length wrong."
            for index in localized_vector:
                data_index = localized_vector[index]
                assert index in self._gm, f"gm misses index #{index}"
                assert isinstance(data_index, np.ndarray) and np.ndim(data_index) == 1, \
                    f"data for element #{index} is not 1-d ndarray"
                assert data_index.shape == (self._gm.num_local_dofs(index), ), \
                    f"data shape of element #{index} does not match the gathering matrix."
            # ====================================================================================
            self._meta_data: Dict = localized_vector
            self._dtype = 'dict'

        elif callable(localized_vector):
            # -------- data check --------------------------------------------------------------
            # ====================================================================================
            self._meta_data = localized_vector
            self._dtype = 'realtime'

        else:
            raise Exception(f"cannot accept data of class {localized_vector.__class__}.")

    @property
    def adjust(self):
        """"""
        return self._adjust

    @property
    def customize(self):
        """"""
        return self._customize

    def ___get_meta_data___(self, index):
        """"""
        if self._dtype == 'dict':
            # noinspection PyUnresolvedReferences
            return self._meta_data[index]
        elif self._dtype == 'realtime':
            return self._meta_data(index)
        else:
            raise NotImplementedError(self._dtype)

    def ___get_adjust_data___(self, index):
        """"""
        if index in self.adjust:
            return self.adjust[index]
        else:
            return self.___get_meta_data___(index)

    def ___get_customize_data___(self, index):
        """"""
        if index in self.customize:
            return self.customize[index]
        else:
            return self.___get_adjust_data___(index)

    def __getitem__(self, index):
        """"""
        return self.___get_customize_data___(index)

    def __len__(self):
        """How many local elements?"""
        return len(self._gm)

    def __iter__(self):
        """iter over all local indices"""
        for index in self._gm:
            yield index

    def __contains__(self, index):
        """If `index` is a valid index of the gathering matrix."""
        return index in self._gm

    def assemble(self):
        """"""
        gm = self._gm
        v = np.zeros(gm.num_dofs)

        for i in self:
            Vi = self[i]  # all adjustments and customizations take effect.
            v[gm[i]] += Vi  # must do this to be consistent with the matrix assembling.

        return Globalize_Static_Vector(v, gm)

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):

            def ___rmul_float_caller___(index):
                A = self.___get_adjust_data___(index)
                return other * A

            return Localize_Static_Vector(___rmul_float_caller___, self._gm)

        else:
            raise NotImplementedError()

    def __add__(self, other):
        """self + other"""

        if other.__class__ is self.__class__:

            assert self._gm == other._gm

            def ___add_caller___(index):
                A = self.___get_adjust_data___(index)
                B = other.___get_adjust_data___(index)
                return A + B

            return Localize_Static_Vector(___add_caller___, self._gm)

        else:
            raise NotImplementedError()

    def __sub__(self, other):
        """self - other"""

        if other.__class__ is self.__class__:

            assert self._gm == other._gm

            def ___sub_caller___(index):
                A = self.___get_adjust_data___(index)
                B = other.___get_adjust_data___(index)
                return A - B

            return Localize_Static_Vector(___sub_caller___, self._gm)

        else:
            raise NotImplementedError()

    def __neg__(self):
        """"""
        def ___neg_caller___(index):
            return - self.___get_adjust_data___(index)
        return Localize_Static_Vector(___neg_caller___, self._gm)


class _Adjust(Frozen):
    """"""

    def __init__(self, V):
        """"""
        self._V = V
        self._adjustments = {}  # keys are cells, values are the data.
        self._freeze()

    def __len__(self):
        return len(self._adjustments)

    def __contains__(self, i):
        """Whether cell #`i` is adjusted?"""
        return i in self._adjustments

    def __getitem__(self, i):
        """Return the instance that contains all adjustments of data for cell #i."""
        return self._adjustments[i]


class _Customize(Frozen):
    """"""

    def __init__(self, V):
        """"""
        self._V = V
        self._customizations = {}  # store the customized data for cells.
        self._freeze()

    def __len__(self):
        return len(self._customizations)

    def __contains__(self, i):
        """Whether cell #`i` is customized?"""
        return i in self._customizations

    def __getitem__(self, i):
        """Return the customized data for cell #i."""
        return self._customizations[i]

    def set_value(self, i, value):
        """Set global value v[i] to be `value`.

        Parameters
        ----------
        i
        value

        Returns
        -------

        """
        elements_local_rows = self._V._gm._find_elements_and_local_indices_of_dofs(i)
        dof = list(elements_local_rows.keys())[0]
        elements, local_rows = elements_local_rows[dof]
        element, local_row = elements[0], local_rows[0]
        data = self._V[element]
        data[local_row] = value
        self._customizations[element] = data
        for element, local_row in zip(elements[1:], local_rows[1:]):
            data = self._V[element]
            data[local_row] = 0
            self._customizations[element] = data

    def set_values(self, global_dofs, cochain):
        """set `v[global_dofs]` to be `cochain`."""
        # first we build up one-2-one relation between dofs and cochain
        if isinstance(cochain, (int, float)):
            cochain = np.ones(len(global_dofs)) * cochain
        else:
            pass

        assert len(global_dofs) == len(cochain), f"len(dofs) != len(cochains)"
        dof_cochain_dict = {}
        for dof, cc in zip(global_dofs, cochain):
            if dof in dof_cochain_dict:
                pass
            else:
                dof_cochain_dict[dof] = cc

        elements_local_rows = self._V._gm._find_elements_and_local_indices_of_dofs(
            list(dof_cochain_dict.keys()))

        involved_elements = list()
        for dof in elements_local_rows:
            for element in elements_local_rows[dof][0]:
                if element not in involved_elements:
                    involved_elements.append(element)

        involved_data = dict()
        for element in involved_elements:
            involved_data[element] = self._V[element]

        for dof in dof_cochain_dict:
            cochain = dof_cochain_dict[dof]
            elements, rows = elements_local_rows[dof]

            element, row = elements[0], rows[0]

            involved_data[element][row] = cochain

            if len(elements) == 1:  # this dof only appear in a single place.
                pass
            else:
                elements, rows = elements[1:], rows[1:]
                for element, row in zip(elements, rows):
                    involved_data[element][row] = 0  # set other places to be 0.

        for element in involved_data:  # update customization
            self._customizations[element] = involved_data[element]


class Localize_Static_Vector_Cochain(Localize_Static_Vector):
    """"""

    def __init__(self, rf, t, localized_vector, gm):
        """"""
        if localized_vector is None:
            pass
        else:
            assert isinstance(localized_vector, dict), \
                f"{Localize_Static_Vector_Cochain} only accepts dict array"
        self._f = rf
        self._t = t
        super().__init__(localized_vector, gm)
        self._freeze()

    def override(self):
        """override `self._data` to be the cochain of `self._f` at time `self._t`."""
        cochain_dict = dict()
        for index in self:
            cochain_dict[index] = self[index]
        self._f[self._t].cochain = cochain_dict
