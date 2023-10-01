# -*- coding: utf-8 -*-
r"""
"""
from typing import Any
from msehy.tools.irregular_gathering_matrix import IrregularGatheringMatrix

import matplotlib.pyplot as plt
from scipy.sparse import isspmatrix_csr, isspmatrix_csc, csr_matrix, issparse
from tools.frozen import Frozen
from msehy.tools.vector.static.local import IrregularStaticLocalVector
from msehy.tools.vector.static.local import IrregularStaticCochainVector

from msehy.tools.matrix.static.assemble import IrregularStaticLocalMatrixAssemble


class IrregularStaticLocalMatrix(Frozen):
    """"""
    def __init__(self, irregular_data: Any, igm_row, igm_col, cache_key=None):
        """"""
        assert igm_row.__class__ is IrregularGatheringMatrix and igm_col.__class__ is IrregularGatheringMatrix, \
            f"row or col gathering matrix class wrong, they must be {IrregularGatheringMatrix}."
        assert len(igm_row) == len(igm_col), f"num cells if gathering matrices do not match."

        if isinstance(irregular_data, (int, float)):
            self._dtype = 'constant'
            self._constant = irregular_data
            self._constant_meta_cache = dict()
            self._cache_key = self._constant_cache_key

        elif (isinstance(irregular_data, dict) or
              (hasattr(irregular_data, '_is_dict_like') and irregular_data._is_dict_like())):

            assert len(irregular_data) == len(igm_row) == len(igm_col), f"length wrong."
            for i in irregular_data:
                data_i = irregular_data[i]
                assert issparse(data_i), f"data for particular cell must be sparse matrix"
                assert data_i.shape == (igm_row.num_local_dofs(i), igm_col.num_local_dofs(i)), \
                    f"data shape for index{i} wrong."
            self._dtype = 'dict'
            self._data = irregular_data

            if cache_key in ('unique', None):
                self._cache_key = self._unique_cache_key
            else:
                self._cache_key = cache_key

        elif callable(irregular_data):
            self._dtype = 'realtime'
            self._data = irregular_data
            assert cache_key is not None, f"when provided callable data, must provide cache_key."
            if cache_key == 'unique':
                self._cache_key = self._unique_cache_key
            else:
                self._cache_key = cache_key

        else:
            raise NotImplementedError(
                f"IrregularStaticLocalMatrix cannot take data of type {irregular_data.__class__}."
            )

        self._gm0_row = igm_row
        self._gm1_col = igm_col
        self._cache = {}
        self._customize = _IrCustomize(self)
        self._adjust = _IrAdjust(self)
        self._assemble = IrregularStaticLocalMatrixAssemble(self)
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        num_cells = len(self._gm0_row)
        self_repr = 'Irregular Static Local Matrix: ' + self._dtype + f"={num_cells}"
        return r"<" + self_repr + super_repr

    def _constant_cache_key(self, i):
        """"""
        assert i in self, f"i={i} is out of range."
        row_shape = len(self._gm0_row[i])
        col_shape = len(self._gm1_col[i])
        return str(row_shape) + '-' + str(col_shape)

    def _unique_cache_key(self, i):
        """"""
        assert i in self, f"i={i} is out of range."
        return 'unique'

    def ___get_meta_data___(self, i):
        """"""
        if self._dtype == 'dict':
            # noinspection PyUnresolvedReferences
            data = self._data[i]
        elif self._dtype == 'constant':
            constant = self._constant
            row_shape = self._gm0_row.num_local_dofs(i)
            col_shape = self._gm1_col.num_local_dofs(i)
            key = str(row_shape) + '-' + str(col_shape)
            if key in self._constant_meta_cache:
                pass
            else:
                shape = (row_shape, col_shape)
                if constant == 0:
                    data = csr_matrix(shape)
                else:
                    raise NotImplementedError()
                self._constant_meta_cache[key] = data

            return self._constant_meta_cache[key]

        elif self._dtype == 'realtime':
            data = self._data(i)
        else:
            raise Exception()

        return data

    def _get_meta_data_from_cache(self, i):
        """"""
        ck = self._cache_key(i)

        if 'unique' in ck:  # we do not cache at all. Use the meta-data.
            return self.___get_meta_data___(i)

        else:  # otherwise, we do dynamic caching.
            assert ck != 'constant',  f"For irregular local matrix, it cannot be constant since shape will change!"
            if ck in self._cache:
                data = self._cache[ck]
            else:
                data = self.___get_meta_data___(i)
                self._cache[ck] = data

            return data

    def _get_data_adjusted(self, i):
        """"""
        if len(self.adjust) == 0:
            data = self._get_meta_data_from_cache(i)
        else:
            if i in self.adjust:
                # adjusted data is cached in `adjust.adjustment` anyway!
                data = self.adjust[i]
            else:
                data = self._get_meta_data_from_cache(i)

        assert isspmatrix_csc(data) or isspmatrix_csr(data), f"data={data.__class__} for cell #{i} is not sparse."

        return data

    def __getitem__(self, i):
        """Get the final (adjusted and customized) matrix for cell #i.
        """
        if i in self.customize:
            data = self.customize[i]
        else:
            data = self._get_data_adjusted(i)

        assert isspmatrix_csc(data) or isspmatrix_csr(data), f"data for cell #i is not sparse."
        shape = (len(self._gm0_row[i]), len(self._gm1_col[i]))
        assert data.shape == shape, f"data shape wrong for index {i}."

        return data

    def __iter__(self):
        """iteration over all cells."""
        for i in self._gm0_row:
            yield i

    def spy(self, i, markerfacecolor='k', markeredgecolor='g', markersize=6):
        """spy the local A of cell #i.

        Parameters
        ----------
        i
        markerfacecolor
        markeredgecolor
        markersize

        Returns
        -------

        """
        M = self[i]
        fig = plt.figure()
        plt.spy(
            M,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markersize=markersize
        )
        plt.tick_params(axis='both', which='major', direction='out')
        plt.tick_params(which='both', top=True, right=True, labelbottom=True, labelright=True)
        plt.show()
        return fig

    @property
    def customize(self):
        """Will not touch the data. Modification will be applied addition to the data.

        Will not touch the dependent matrices. See ``adjust``.
        """
        return self._customize

    @property
    def adjust(self):
        """Will touch the data (not the meta-data).
        Modification will be applied to new copies of the meta-data.

        Adjustment will change the matrices dependent on me. For example, B = A.T. If I adjust A late on,
        B will also change.

        While if we ``customize`` A, B will not be affected.
        """
        return self._adjust

    @property
    def assemble(self):
        """assemble self."""
        return self._assemble

    def __contains__(self, i):
        """if cell #i is valid."""
        return i in self._gm0_row

    @property
    def num_cells(self):
        """How many cells?"""
        return len(self._gm0_row)

    def __len__(self):
        """How many cells?"""
        return self.num_cells

    @staticmethod
    def is_static():
        """static"""
        return True

    @property
    def T(self):
        """The `customization` will not have an effect. The adjustment will be taken into account."""
        return self.__class__(
            self._data_T,
            self._gm1_col,
            self._gm0_row,
            cache_key=self._cache_key_T,
        )

    def _data_T(self, i):
        """"""
        data = self._get_data_adjusted(i)
        return data.T

    def _cache_key_T(self, i):
        """"""
        if i in self.adjust:
            return 'unique'
        else:
            return self._cache_key(i)

    def __matmul__(self, other):
        """self @ other.

        The `customization` of both entries will not have an effect. The adjustments will be taken into account.
        """
        if other.__class__ is IrregularStaticLocalMatrix:

            _matmul = _MatmulMatMat(self, other)
            cache_key = _matmul.cache_key

            static = self.__class__(_matmul, self._gm0_row, other._gm1_col, cache_key=cache_key)

            return static

        elif other.__class__ in (IrregularStaticLocalVector, IrregularStaticCochainVector):

            _matmul = _mat_mul_mat_vec(self, other)

            assert _matmul is not None, f"for @, we do not accept None data."

            return IrregularStaticLocalVector(_matmul, self._gm0_row)

        else:
            raise NotImplementedError(f"{other.__class__}.")

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            factor_mat = _FactorRmul(other, self)
            return self.__class__(
                factor_mat, self._gm0_row, self._gm1_col, cache_key=factor_mat._cache_key
            )
        else:
            raise Exception()

    def __neg__(self):
        """- self"""
        return self.__class__(
            self._data_neg,
            self._gm0_row,
            self._gm1_col,
            cache_key=self._cache_key_T,
        )

    def __add__(self, other):
        """self + other"""
        if other.__class__ is self.__class__:
            # self + another one of the same class
            add_sub_helper = _AddSubHelper(self, other, '+')
            return self.__class__(
                add_sub_helper, self._gm0_row, self._gm1_col, cache_key=add_sub_helper._cache_key
            )
        else:
            raise NotImplementedError(f"cannot + {other}")

    def __sub__(self, other):
        """self - other"""
        if other.__class__ is self.__class__:
            # self + another one of the same class
            add_sub_helper = _AddSubHelper(self, other, '-')
            return self.__class__(
                add_sub_helper, self._gm0_row, self._gm1_col, cache_key=add_sub_helper._cache_key
            )
        else:
            raise NotImplementedError(f"cannot + {other}")

    def _data_neg(self, i):
        return - self._get_data_adjusted(i)


class _AddSubHelper(Frozen):
    """"""
    def __init__(self, m0, m1, plus_or_minus):
        """"""
        self._m0 = m0
        self._m1 = m1
        assert plus_or_minus in ('+', '-')
        self._plus_or_minus = plus_or_minus
        self._freeze()

    def __call__(self, i):
        """data for cell #i"""
        d0 = self._m0._get_data_adjusted(i)
        d1 = self._m1._get_data_adjusted(i)
        if self._plus_or_minus == '+':
            return d0 + d1
        elif self._plus_or_minus == '-':
            return d0 - d1
        else:
            raise Exception()

    def _cache_key(self, i):
        """"""
        if i in self._m0.adjust or i in self._m1.adjust:
            return 'unique'
        else:
            ck0 = self._m0._cache_key(i)
            ck1 = self._m1._cache_key(i)
            return ck0+'@'+ck1


class _FactorRmul(Frozen):
    """"""
    def __init__(self, factor, mat):
        """"""
        self._f = factor
        self._mat = mat
        self._freeze()

    def __call__(self, i):
        """"""
        mat = self._mat._get_data_adjusted(i)
        return self._f * mat

    def _cache_key(self, i):
        """"""
        if i in self._mat.adjust:
            return 'unique'
        else:
            return self._mat._cache_key(i)


class _MatmulMatMat(Frozen):
    """A @ B, A and A are both irregular static local matrix."""

    def __init__(self, M0, M1):
        """"""
        self._m0 = M0
        self._m1 = M1
        self._freeze()

    def __call__(self, i):
        """"""
        data0 = self._m0._get_data_adjusted(i)
        data1 = self._m1._get_data_adjusted(i)
        return data0 @ data1

    def cache_key(self, i):
        """"""
        if i in self._m0.adjust:
            return 'unique'
        else:
            ck0 = self._m0._cache_key(i)

        if i in self._m1.adjust:
            return 'unique'
        else:
            ck1 = self._m1._cache_key(i)

        if 'unique' in (ck0, ck1):
            return 'unique'
        else:
            return ck0+'@'+ck1


def _mat_mul_mat_vec(m, v):
    """
    a matrix @ a vector

    Parameters
    ----------
    m
    v

    Returns
    -------

    """
    vec = v.data  # make sure the 2D data is ready; a dictionary.

    # if isinstance(v, MsePyRootFormStaticCochainVector):
    #     print(v._t)

    if vec is None:
        raise Exception('static local vector has no data, cannot @ it.')
    else:
        assert isinstance(vec, dict), f"data of irregular static local vector must be a dictionary."

    if len(m.adjust) == 0:  # The matrix does not have any adjustment.

        if m._dtype in ('constant', 'dict', 'realtime'):

            data = dict()
            for i in m:
                m_cell = m[i]
                data[i] = m_cell @ vec[i]

        else:
            raise NotImplementedError(f"{m._dtype}")

    else:
        raise NotImplementedError(f"take adjusted data.")

    return data


class _IrAdjust(Frozen):
    """"""

    def __init__(self, M):
        """"""
        self._M = M
        self._adjustments = {}  # keys are cells, values are the data.
        self._freeze()

    def __len__(self):
        return len(self._adjustments)

    def __contains__(self, i):
        """Whether cell #i is adjusted?"""
        return i in self._adjustments

    def __getitem__(self, i):
        """Return the instance that contains all adjustments of data for cell #i."""
        return self._adjustments[i]


class _IrCustomize(Frozen):
    """"""

    def __init__(self, M):
        """"""
        self._M = M
        self._customizations = {}  # store the customized data for cells.
        self._freeze()

    def __len__(self):
        return len(self._customizations)

    def __contains__(self, i):
        """Whether cell #i is customized?"""
        return i in self._customizations

    def __getitem__(self, i):
        """Return the customized data for cell #i."""
        return self._customizations[i]

    def clear(self, i=None):
        """clear customizations for cell #i.

        When `i` is None, clear for all cells.
        """
        if i is None:
            self._customizations = {}
        else:
            raise NotImplementedError()

    def identify_row(self, i):
        """identify global row #i: M[i,:] = 0 and M[i, i] = 1, where M means the assembled matrix.

        Parameters
        ----------
        i

        Returns
        -------

        """
        cells_local_rows = self._M._gm0_row._find_fundamental_cells_and_local_indices_of_dofs(i)

        dof = list(cells_local_rows.keys())[0]
        cells, local_rows = cells_local_rows[dof]
        assert len(cells) == len(local_rows), f"something is wrong!"

        cell, local_row = cells[0], local_rows[0]  # indentify in the first place

        data = self._M[cell].copy().tolil()
        data[local_row, :] = 0
        data[local_row, local_row] = 1
        self._customizations[cell] = data.tocsr()

        for cell, local_row in zip(cells[1:], local_rows[1:]):  # zero rows in other places.
            data = self._M[cell].copy().tolil()
            data[local_row, :] = 0
            self._customizations[cell] = data.tocsr()

    def identify_diagonal(self, global_dofs):
        """Set the global rows of ``global_dofs`` to be all zero except the diagonal to be 1."""

        elements_local_rows = self._M._gm0_row._find_fundamental_cells_and_local_indices_of_dofs(global_dofs)

        set_0_elements_rows = dict()
        set_1_elements_rows = dict()

        for global_dof in elements_local_rows:
            elements, local_rows = elements_local_rows[global_dof]
            representative_element, representative_row = elements[0], local_rows[0]

            if representative_element in set_1_elements_rows:
                pass
            else:
                set_1_elements_rows[representative_element] = list()
            set_1_elements_rows[representative_element].append(representative_row)

            other_elements, other_rows = elements[1:], local_rows[1:]
            for k, oe in enumerate(other_elements):
                if oe in set_0_elements_rows:
                    pass
                else:
                    set_0_elements_rows[oe] = list()

                set_0_elements_rows[oe].append(
                    other_rows[k]
                )

        all_involved_elements = set(list(set_0_elements_rows.keys()) + list(set_1_elements_rows.keys()))
        for element in all_involved_elements:
            data = self._M[element].copy().tolil()

            if element in set_0_elements_rows:
                zero_rows = set_0_elements_rows[element]
                data[zero_rows, :] = 0

            else:
                pass

            if element in set_1_elements_rows:
                one_rows = set_1_elements_rows[element]
                data[one_rows, :] = 0
                data[one_rows, one_rows] = 1

            else:
                pass

            self._customizations[element] = data.tocsr()

    def set_zero(self, global_dofs):
        """Set the global rows of ``global_dofs`` to be all zero."""
        elements_local_rows = self._M._gm0_row._find_fundamental_cells_and_local_indices_of_dofs(global_dofs)
        set_0_elements_rows = dict()

        for global_dof in elements_local_rows:
            elements, local_rows = elements_local_rows[global_dof]
            for k, oe in enumerate(elements):
                if oe in set_0_elements_rows:
                    pass
                else:
                    set_0_elements_rows[oe] = list()

                set_0_elements_rows[oe].append(
                    local_rows[k]
                )

        for element in set_0_elements_rows:
            data = self._M[element].copy().tolil()
            if element in set_0_elements_rows:
                zero_rows = set_0_elements_rows[element]
                data[zero_rows, :] = 0

            else:
                pass
            self._customizations[element] = data.tocsr()
