# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from numpy import diff
from scipy.sparse import bmat as sp_bmat
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc

from phyem.tools.frozen import Frozen
from phyem.msepy.mesh.elements.main import _DataDictDistributor
from phyem.msepy.tools.vector.static.local import MsePyStaticLocalVector
from phyem.msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector

from phyem.msepy.tools.gathering_matrix import RegularGatheringMatrix
from phyem.msepy.tools.matrix.static.assembled import MsePyStaticAssembledMatrix


class MsePyStaticLocalMatrix(Frozen):
    """"""
    def __init__(self, data, gm_row, gm_col, cache_key=None):
        """"""
        if data.__class__ is _DataDictDistributor:
            self._dtype = 'ddd'
            self._data = data  # element-wise csc or csr matrix.
            self._cache_key = data._cache_key_generator

        elif issparse(data) or data == 0:   # constant data; same in all elements.
            if issparse(data):
                if not (isspmatrix_csc(data) or isspmatrix_csr(data)):
                    data = data.tocsr()
                else:
                    pass
                shape0, shape1 = data.shape  # must be regular gathering matrix, so `.shape` does not raise Error.
                assert shape0 == gm_row.shape[1], f"row shape wrong"
                assert shape1 == gm_col.shape[1], f"col shape wrong"
            elif data == 0:
                shape0 = gm_row.shape[1]
                shape1 = gm_col.shape[1]
                data = csr_matrix((shape0, shape1))
            else:
                raise Exception()
            self._dtype = 'constant'
            self._data = data
            self._cache_key = self._constant_cache_key

        elif callable(data):
            self._dtype = 'realtime'
            self._data = data
            assert cache_key is not None, f"when provided callable data, must provide cache_key."
            if cache_key == 'unique':
                self._cache_key = self._unique_cache_key
            else:
                self._cache_key = cache_key

        elif isinstance(data, dict):
            self._dtype = 'realtime'
            self.___dict_2_realtime_data___ = data
            self._data = self._dict_2_realtime_caller
            assert cache_key is None, f"when provided dict, provide no cache key."
            self._cache_key = self._unique_cache_key

        else:
            raise NotImplementedError(f"MsePyLocalMatrix cannot take data of type {data.__class__}.")

        self._gm0_row = gm_row
        self._gm1_col = gm_col
        self._constant_cache = None
        self._cache = {}
        self._irs = None
        self._customize = _MsePyStaticLocalMatrixCustomize(self)
        self._assemble = MsePyStaticLocalMatrixAssemble(self)
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        shape = self[0].shape
        num_elements = self._gm0_row.num_elements
        self_repr = 'MsePy Static Local Matrix: ' + self._dtype + f"={num_elements}+{shape}"
        return r"<" + self_repr + super_repr

    def _dict_2_realtime_caller(self, i):
        """"""
        return self.___dict_2_realtime_data___[i]

    def _constant_cache_key(self, i):
        """"""
        assert i in self, f"i={i} is out of range."
        return 'constant'

    def _unique_cache_key(self, i):
        """"""
        assert i in self, f"i={i} is out of range."
        return 'unique'

    def ___get_meta_data___(self, i):
        """"""
        if self._dtype == 'ddd':
            # noinspection PyUnresolvedReferences
            data = self._data.get_data_of_element(i)
        elif self._dtype == 'constant':
            data = self._data
        elif self._dtype == 'realtime':
            data = self._data(i)
        else:
            raise Exception()

        return data

    def _get_meta_data_from_cache(self, i):
        """"""
        ck = self._cache_key(i)
        if ck == 'constant':  # of course, we try to return from the constant cache.
            if self._constant_cache is None:
                self._constant_cache = self.___get_meta_data___(i)
            else:
                pass
            return self._constant_cache

        elif 'unique' in ck:  # we do not cache at all. Use the meta-data.
            return self.___get_meta_data___(i)

        else:  # otherwise, we do dynamic caching.
            if ck in self._cache:
                data = self._cache[ck]
            else:
                data = self.___get_meta_data___(i)
                self._cache[ck] = data

            return data

    def __getitem__(self, i):
        """Get the final (adjusted and customized) matrix for element #i.
        """
        if i in self.customize:
            data = self.customize[i]
        else:
            data = self._get_meta_data_from_cache(i)

        assert isspmatrix_csc(data) or isspmatrix_csr(data), f"data for element #i is not sparse."

        return data

    def __iter__(self):
        """iteration over all elements."""
        for i in range(self.num_elements):
            yield i

    def spy(self, i, markerfacecolor='k', markeredgecolor='g', markersize=6):
        """spy the local A of element #i.

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
    def assemble(self):
        """assemble self."""
        return self._assemble

    def __contains__(self, i):
        """if element #i is valid."""
        return i in range(self.num_elements)

    @property
    def num_elements(self):
        """How many elements?"""
        return self._gm0_row.num_elements

    def __len__(self):
        """How many elements?"""
        return self.num_elements

    @staticmethod
    def is_static():
        """static"""
        return True

    def is_untouched(self):
        """If no customization, it is untouched."""
        return True if len(self.customize) == 0 else False

    def _is_regularly_square(self):
        if self._irs is None:
            self._irs = np.all(self._gm0_row._gm == self._gm1_col._gm)
        return self._irs

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
        assert len(self.customize) == 0, f"customized matrix bears no operations."
        data = self[i]
        return data.T

    def _cache_key_T(self, i):
        """"""
        return self._cache_key(i)

    def __matmul__(self, other):
        """self @ other.

        The `customization` of both entries will not have an effect. The adjustments will be taken into account.
        """
        if other.__class__ is MsePyStaticLocalMatrix:

            _matmul = _MatmulMatMat(self, other)
            cache_key = _matmul.cache_key

            static = self.__class__(_matmul, self._gm0_row, other._gm1_col, cache_key=cache_key)

            return static

        elif other.__class__ in (MsePyStaticLocalVector, MsePyRootFormStaticCochainVector):

            _matmul = _mat_mul_mat_vec(self, other)

            assert _matmul is not None, f"for @, we do not accept None data."

            return MsePyStaticLocalVector(_matmul, self._gm0_row)

        elif other.__class__ is np.ndarray and other.ndim == 2:
            # this 2d array saves as a local cochain.
            assert self.num_elements == other.shape[0], f"2d array axis[0] length must be equal to element amount."
            assert self._gm1_col.num_local_dofs == other.shape[1], \
                f"2d array axis[1] length must be equal to col local dofs."

            vec = MsePyStaticLocalVector(other, self._gm1_col)
            # noinspection PyTypeChecker
            return self @ vec

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
            self._neg_T,
            self._gm0_row,
            self._gm1_col,
            cache_key=self._cache_key_T,
        )

    def _neg_T(self, i):
        assert len(self.customize) == 0, f"customized matrix bears no operation."
        return - self[i]

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
        """data for element #i"""
        assert len(self._m0.customize) == 0 and len(self._m1.customize) == 0, \
            f"customized matrix bears no operation."
        d0 = self._m0[i]
        d1 = self._m1[i]
        if self._plus_or_minus == '+':
            return d0 + d1
        elif self._plus_or_minus == '-':
            return d0 - d1
        else:
            raise Exception()

    def _cache_key(self, i):
        """"""
        ck0 = self._m0._cache_key(i)
        ck1 = self._m1._cache_key(i)
        if ck0 == 'constant' and ck1 == 'constant':
            return 'constant'
        else:
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
        assert len(self._mat.customize) == 0, f"customized matrix bears no operation."
        mat = self._mat[i]
        return self._f * mat

    def _cache_key(self, i):
        """"""
        return self._mat._cache_key(i)


class _MatmulMatMat(Frozen):
    """"""

    def __init__(self, M0, M1):
        """"""
        self._m0 = M0
        self._m1 = M1
        self._freeze()

    def __call__(self, i):
        """"""
        assert len(self._m0.customize) == 0 and len(self._m1.customize) == 0, \
            f"customized matrix bears no operation."
        data0 = self._m0[i]
        data1 = self._m1[i]
        return data0 @ data1

    def cache_key(self, i):
        """"""
        ck0 = self._m0._cache_key(i)
        ck1 = self._m1._cache_key(i)
        if 'unique' in (ck0, ck1):
            return 'unique'
        elif ck0 == 'constant' and ck1 == 'constant':
            return 'constant'
        else:
            return ck0+'@'+ck1


def _mat_mul_mat_vec(m, v):
    """"""

    vec = v.data  # make sure the 2D data is ready.

    if vec is None:
        raise Exception('msepy local vector has no data, cannot @ it.')
    else:
        pass

    if len(m.customize) == 0:
        if m._dtype == 'constant':

            mat = m._data.toarray()

            data = np.einsum('ij, ej -> ei', mat, vec, optimize='optimal')

        elif m._dtype == 'ddd':

            mat = m._data

            vec = mat.split(vec)

            data = list()

            for ci in mat.cache_indices:

                mat_ci = mat.get_data_of_cache_index(ci)

                if issparse(mat_ci):
                    mat_ci = mat_ci.toarray()

                elif isinstance(mat_ci, np.ndarray):
                    assert np.ndim(mat_ci) == 2, f"must be a 2d ndarray."

                else:
                    raise NotImplementedError('data type not accepted!')

                vec_ci = vec[ci]

                data.append(
                    np.einsum('ij, ej -> ei', mat_ci, vec_ci, optimize='optimal')
                )

            data = mat.merge(data)

        elif m._dtype == 'realtime':

            data = list()
            for e in m:
                m_element_e = m[e]
                data.append(
                    m_element_e @ vec[e]
                )
            data = np.vstack(data)

        else:
            raise NotImplementedError(f"{m._dtype}")

        return data

    else:
        def ___mat_at_vec___(i):
            """"""
            mat_i = m[i]
            vec_i = vec[i]
            return mat_i @ vec_i

        return ___mat_at_vec___


class _MsePyStaticLocalMatrixCustomize(Frozen):
    """"""

    def __init__(self, M):
        """"""
        self._M = M
        self._customizations = {}  # store the customized data for elements.
        self._freeze()

    def __len__(self):
        return len(self._customizations)

    def __contains__(self, i):
        """Whether element #i is customized?"""
        return i in self._customizations

    def __getitem__(self, i):
        """Return the customized data for element #i."""
        return self._customizations[i]

    def clear(self, i=None):
        """clear customizations for element #i.

        When `i` is None, clear for all elements.
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
        assert self._M._is_regularly_square(), f"need a regularly square matrix."
        elements_local_rows = self._M._gm0_row._find_elements_and_local_indices_of_dofs(i)

        dof = list(elements_local_rows.keys())[0]
        elements, local_rows = elements_local_rows[dof]
        assert len(elements) == len(local_rows), f"something is wrong!"

        element, local_row = elements[0], local_rows[0]  # indentify in the first place

        data = self._M[element].copy().tolil()
        data[local_row, :] = 0
        data[local_row, local_row] = 1
        self._customizations[element] = data.tocsr()

        for element, local_row in zip(elements[1:], local_rows[1:]):  # zero rows in other places.
            data = self._M[element].copy().tolil()
            data[local_row, :] = 0
            self._customizations[element] = data.tocsr()

    def identify_diagonal(self, global_dofs):
        """Set the global rows of ``global_dofs`` to be all zero except the diagonal to be 1."""
        assert self._M._is_regularly_square(), f"need a regularly square matrix."

        elements_local_rows = self._M._gm0_row._find_elements_and_local_indices_of_dofs(global_dofs)

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

    def set_values_and_zero_rest(self, rows, cols, vals):
        """Let self be `M`. We do

            M[rows] = 0
            M[rows, cols] = values

        rows and cols are both global.

        Parameters
        ----------
        rows
        cols
        vals

        Returns
        -------

        """
        self.set_zero(rows)
        self.set_values(rows, cols, vals)

    def set_values(self, rows, cols, vals):
        """Let self be M. We do
            M[rows, cols] = vals

        where rows and cols are both global.

        Parameters
        ----------
        rows
        cols
        vals

        Returns
        -------

        """
        if isinstance(rows, int):
            rows = [rows]
        else:
            pass
        if isinstance(cols, int):
            cols = [cols]
        else:
            pass
        if isinstance(vals, (int, float)):
            vals = np.array(
                [vals for _ in range(len(rows))]
            )
        else:
            pass

        assert isinstance(rows, (list, tuple)), f"rows must be a list or tuple."
        assert isinstance(cols, (list, tuple)), f"cols must be a list or tuple."
        assert isinstance(vals, (list, tuple, np.ndarray)), f"vals must be a list or tuple or ndarray."
        assert np.ndim(rows) == np.ndim(cols) == np.ndim(vals) == 1, f"data must be 1d."
        assert len(rows) == len(cols) == len(vals), f"data shape must be same."

        elements_local_rows = self._M._gm0_row._find_elements_and_local_indices_of_dofs(rows)
        elements_local_cols = self._M._gm1_col._find_elements_and_local_indices_of_dofs(cols)

        all_positions = dict()

        for global_row, global_col, val in zip(elements_local_rows, elements_local_cols, vals):
            row_elements_indices = elements_local_rows[global_row]
            col_elements_indices = elements_local_cols[global_col]

            row_elements, row_indices = row_elements_indices
            col_elements, col_indices = col_elements_indices

            len_row_elements = len(row_elements)
            len_col_elements = len(col_elements)

            if len_row_elements == len_col_elements == 1:

                re = row_elements[0]
                ce = col_elements[0]

                if re == ce:  # if we reach here, we are probably dealing boundary dofs.
                    element = re
                    ri = row_indices[0]
                    ci = col_indices[0]

                    if element in all_positions:
                        pass
                    else:
                        all_positions[element] = list()

                    all_positions[element].append((ri, ci, val))

                else:
                    raise Exception(
                        'dof elements are not consistent. It is difficult.')

            else:
                raise Exception(
                    f'dof location is not unique {len_row_elements} {len_col_elements}. '
                    f'It is difficult.'
                )

        for element in all_positions:
            data = self._M[element].copy().tolil()
            for ri_ci_val in all_positions[element]:
                ri, ci, val = ri_ci_val
                data[ri, ci] = val
            self._customizations[element] = data.tocsr()

    def set_zero(self, global_dofs):
        """Set the global rows of ``global_dofs`` to be all zero."""
        elements_local_rows = self._M._gm0_row._find_elements_and_local_indices_of_dofs(global_dofs)
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


def bmat(A_2d_list):
    """"""
    row_shape = len(A_2d_list)
    for Ai_ in A_2d_list:
        assert isinstance(Ai_, (list, tuple)), f"bmat must apply to 2d list or tuple."
    col_shape = len(A_2d_list[0])

    row_gms = [None for _ in range(row_shape)]
    col_gms = [None for _ in range(col_shape)]

    for i in range(row_shape):
        for j in range(col_shape):
            A_ij = A_2d_list[i][j]

            if A_ij is None:
                pass
            else:
                assert A_ij.__class__ is MsePyStaticLocalMatrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"
                row_gm_i = A_ij._gm0_row
                col_gm_j = A_ij._gm1_col

                if row_gms[i] is None:
                    row_gms[i] = row_gm_i
                else:
                    assert row_gms[i] is row_gm_i, f"by construction, this must be the case as we only construct" \
                                                   f"gathering matrix once and store only once copy somewhere!"

                if col_gms[j] is None:
                    col_gms[j] = col_gm_j
                else:
                    assert col_gms[j] is col_gm_j, f"by construction, this must be the case as we only construct" \
                                                   f"gathering matrix once and store only once copy somewhere!"

    chain_row_gm = RegularGatheringMatrix(row_gms)
    chain_col_gm = RegularGatheringMatrix(col_gms)

    # only adjustments take effect. Customization will be skipped.
    M = _MsePyStaticLocalMatrixBmat(A_2d_list, (row_shape, col_shape))

    return MsePyStaticLocalMatrix(M, chain_row_gm, chain_col_gm, M.cache_key)


class _MsePyStaticLocalMatrixBmat(Frozen):
    """"""

    def __init__(self, A_2d_list, shape):
        """"""
        self._A = A_2d_list
        self._shape = shape
        self._freeze()

    def __call__(self, i):
        row_shape, col_shape = self._shape
        data = [[None for _ in range(col_shape)] for _ in range(row_shape)]
        for r in range(row_shape):
            for c in range(col_shape):
                Arc = self._A[r][c]
                if Arc is None:
                    pass
                else:
                    data[r][c] = Arc[i]  # All customizations take effect!

        return sp_bmat(
            data, format='csr'
        )

    def cache_key(self, i):
        """Do this in real time."""
        row_shape, col_shape = self._shape
        keys = list()
        for r in range(row_shape):
            for c in range(col_shape):
                Arc = self._A[r][c]

                if Arc is None:
                    pass
                else:
                    if i in Arc.customize:  # important! do not use adjust!
                        return 'unique'
                    else:
                        key = Arc._cache_key(i)
                        if key == 'unique':
                            return 'unique'
                        else:
                            keys.append(
                                key
                            )

        if all([_ == 'constant' for _ in keys]):
            return 'constant'
        else:
            return ''.join(keys)


_msepy_assembled_StaticMatrix_cache = {}
# we can cache the assembled matrices in case that it is the same for many or even all time steps.


class MsePyStaticLocalMatrixAssemble(Frozen):
    """"""

    @property
    def ___assembled_class___(self):
        return MsePyStaticAssembledMatrix

    def __init__(self, M):
        """"""
        self._M = M
        self._freeze()

    def __call__(self, format='csc', cache=None):
        """

        Parameters
        ----------
        format
        cache :
            We can manually cache the assembled matrix by set ``cache`` to be a string. When next time
            it sees the same `cache` it will return the cached matrix from the cache, i.e.,
            ``_msepy_assembled_StaticMatrix_cache``.

        Returns
        -------

        """
        if cache is not None:
            assert isinstance(cache, str), f"cache must a string."
            if cache in _msepy_assembled_StaticMatrix_cache:
                return _msepy_assembled_StaticMatrix_cache[cache]
            else:
                pass

        else:
            pass

        gm_row = self._M._gm0_row
        gm_col = self._M._gm1_col

        dep = int(gm_row.num_dofs)
        wid = int(gm_col.num_dofs)

        ROW = list()
        COL = list()
        DAT = list()

        if format == 'csc':
            SPA_MATRIX = csc_matrix
        elif format == 'csr':
            SPA_MATRIX = csr_matrix
        else:
            raise Exception

        # A = SPA_MATRIX((dep, wid))  # initialize a sparse matrix

        for i in self._M:

            Mi = self._M[i]  # all adjustments and customizations take effect
            indices = Mi.indices
            indptr = Mi.indptr
            data = Mi.data
            nums: list = list(diff(indptr))
            row = []
            col = []

            if Mi.__class__.__name__ == 'csc_matrix':
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    row.extend(gm_row[i][idx])
                    col.extend([gm_col[i][j] for _ in range(num)])

            elif Mi.__class__.__name__ == 'csr_matrix':
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    row.extend([gm_row[i][j] for _ in range(num)])
                    col.extend(gm_col[i][idx])

            else:
                raise Exception("I can not handle %r." % Mi)

            ROW.extend(row)
            COL.extend(col)
            DAT.extend(data)

            # if len(DAT) > 1e7:  # every 10 million data, we make it into a sparse matrix.
            #     _ = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))  # we make it into sparse
            #
            #     del ROW, COL, DAT
            #     A += _
            #     del _
            #     ROW = list()
            #     COL = list()
            #     DAT = list()

        # _ = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))  # we make it into sparse
        # del ROW, COL, DAT
        # A += _

        A = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))
        A = self.___assembled_class___(A, gm_row, gm_col)
        if isinstance(cache, str):
            _msepy_assembled_StaticMatrix_cache[cache] = A
        return A
