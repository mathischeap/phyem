# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK
import matplotlib.pyplot as plt
from tools.frozen import Frozen
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix
from scipy.sparse import isspmatrix_csr, isspmatrix_csc, csc_matrix, csr_matrix
from msehtt.static.form.cochain.instant import MseHttTimeInstantCochain
from msehtt.static.form.cochain.vector.static import MseHttStaticCochainVector
from msehtt.tools.vector.static.local import MseHttStaticLocalVector


class MseHttStaticLocalMatrix(Frozen):
    """"""
    def __init__(self, data, gm_row, gm_col, cache_key=None):
        """"""
        assert gm_row.__class__ is MseHttGatheringMatrix, f"gm row class wrong."
        assert gm_col.__class__ is MseHttGatheringMatrix, f"gm col class wrong."
        assert len(gm_row) == len(gm_col), f"gm length dis-match."
        for i in gm_row:
            assert i in gm_col, f"element #{i} not find in gm col."
        self._gm_row = gm_row
        self._gm_col = gm_col
        self._receive_data(data)
        self._parse_cache_key(cache_key)
        self._assemble = MseHttStaticLocalMatrixAssemble(self)
        self._freeze()

    def _receive_data(self, data):
        """"""
        if callable(data):
            self._dtype = 'realtime'
            self._data = data

        elif isinstance(data, dict):
            self._dtype = 'dict'
            for i in data:
                assert i in self._gm_row, f"element #{i} is not a rank element."
                di = data[i]
                assert isspmatrix_csr(di) or isspmatrix_csc(di), f"data for element #{i} is not csr or csc."
                shape_row, shape_col = di.shape
                assert shape_row == self._gm_row.num_local_dofs(i), f"row shape wrong for element #{i}."
                assert shape_col == self._gm_col.num_local_dofs(i), f"col shape wrong for element #{i}."
            for i in self._gm_row:
                if i not in data:
                    assert self._gm_row.num_local_dofs(i) == 0 or self._gm_col.num_local_dofs(i) == 0
                else:
                    pass
            self._data = data
        elif isinstance(data, (int, float)) and data == 0:
            self._dtype = 'dict'
            self._data = {}

        else:
            raise NotImplementedError(f"MseHtt LocalMatrix cannot take data of type {data.__class__}.")

        self._customize = None
        self._cache = {}

    def _parse_cache_key(self, cache_key):
        """"""
        if cache_key is None:
            self._cache_key = self._unique_cache_key
        elif cache_key == 'unique':
            self._cache_key = self._unique_cache_key
        elif isinstance(cache_key, dict):
            self.___cache_key_dict___ = cache_key
            self._cache_key = self._dict_cache_key_caller
        elif callable(cache_key):
            self._cache_key = cache_key
        elif cache_key == 'zero':
            self._cache_key = self.___zero_cache_key___
        else:
            raise NotImplementedError()

        for i in self:
            _ = self._cache_key(i)  # make sure cache key is valid for all rank great elements.

    def ___zero_cache_key___(self, i):
        """"""
        return str((self._gm_row.num_local_dofs(i), self._gm_col.num_local_dofs(i)))

    def _unique_cache_key(self, i):
        """"""
        assert i in self, f"i={i} is out of range."
        return 'unique'

    def _dict_cache_key_caller(self, i):
        if i in self.___cache_key_dict___:
            return self.___cache_key_dict___[i]
        else:
            return str((self._gm_row.num_local_dofs(i), self._gm_col.num_local_dofs(i)))

    @property
    def customize(self):
        """"""
        if self._customize is None:
            # noinspection PyAttributeOutsideInit
            self._customize = _Static_LocalMatrix_Customize(self)
        return self._customize

    @property
    def assemble(self):
        """assemble self."""
        return self._assemble

    def _get_meta_data(self, i):
        """"""
        if self._dtype == 'realtime':
            return self._data(i)
        elif self._dtype == 'dict':
            if i in self._data:
                # noinspection PyUnresolvedReferences
                return self._data[i]
            else:
                return csc_matrix((self._gm_row.num_local_dofs(i), self._gm_col.num_local_dofs(i)))
        else:
            raise NotImplementedError()

    def __getitem__(self, i):
        """"""
        assert i in self, f"element #{i} is not a rank element."
        if i in self.customize:
            data = self.customize[i]
        else:
            cache_key = self._cache_key(i)
            if 'unique' in cache_key:
                data = self._get_meta_data(i)
            else:
                if cache_key in self._cache:
                    data = self._cache[cache_key]
                else:
                    data = self._get_meta_data(i)
                    self._cache[cache_key] = data
        assert isspmatrix_csr(data) or isspmatrix_csc(data), f"must be csc or csr!"
        return data

    def spy(self, i, markerfacecolor='k', markeredgecolor='g', markersize=6):
        """spy the local A of rank element #i.

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

    def __contains__(self, i):
        """"""
        return i in self._gm_row

    def __len__(self):
        """"""
        return len(self._gm_row)

    def __iter__(self):
        """"""
        for i in self._gm_row:
            yield i

    def __neg__(self):
        """"""
        def data_caller(e):
            return - self[e]

        return self.__class__(data_caller, self._gm_row, self._gm_col, cache_key=self._cache_key)

    @property
    def T(self):
        """"""
        def data_caller(e):
            return (self[e]).T

        return self.__class__(data_caller, self._gm_col, self._gm_row, cache_key=self._cache_key)

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):

            def data_caller(i):
                return other * self[i]

            return self.__class__(data_caller, self._gm_row, self._gm_col, cache_key=self._cache_key)

        else:
            raise NotImplementedError()

    def __add__(self, other):
        """self + other"""
        if other.__class__ is self.__class__:
            def data_caller(i):
                return self[i] + other[i]

            def cache_key_caller(i):
                key1 = self._cache_key(i)
                key2 = other._cache_key(i)
                if key1 == 'unique' or key2 == 'unique':
                    return 'unique'
                else:
                    return key1 + '.' + key2
            return self.__class__(data_caller, self._gm_row, self._gm_col, cache_key=cache_key_caller)

    def __matmul__(self, other):
        """"""
        if other.__class__ is MseHttTimeInstantCochain:
            f = {}
            for e in self:
                M = self[e]
                v = other[e]
                f[e] = M @ v
            return MseHttStaticLocalVector(f, self._gm_row)

        elif other.__class__ is self.__class__:

            def data_caller(i):
                return self[i] @ other[i]

            def cache_key_caller(i):
                key1 = self._cache_key(i)
                if key1 == 'unique':
                    return 'unique'
                else:
                    key2 = other._cache_key(i)
                    if key2 == 'unique':
                        return 'unique'
                    else:
                        return key1 + key2

            return self.__class__(data_caller, self._gm_row, other._gm_col, cache_key=cache_key_caller)

        elif other.__class__ is MseHttStaticCochainVector:

            def data_caller(i):
                return self[i] @ other[i]

            return MseHttStaticLocalVector(data_caller, self._gm_row)

        elif other.__class__ is MseHttStaticLocalVector:

            def data_caller(i):
                return self[i] @ other[i]

            return MseHttStaticLocalVector(data_caller, self._gm_row)

        else:
            raise NotImplementedError(other.__class__)


class _Static_LocalMatrix_Customize(Frozen):
    """"""
    def __init__(self, mat):
        """"""
        self._mat = mat
        self._customizations = {}
        self._freeze()

    def __len__(self):
        """"""
        return len(self._customizations)

    def __contains__(self, item):
        """"""
        return item in self._customizations

    def __getitem__(self, i):
        """"""
        return self._customizations[i]

    def clear(self, i=None):
        """clear customizations for element #i.

        When `i` is None, clear for all elements.
        """
        if i is None:
            self._customizations = {}
        else:
            raise NotImplementedError()

    def zero_row(self, i):
        """identify global row #i: M[i,:] = 0 and M[i, i] = 1, where M means the assembled matrix.

        Parameters
        ----------
        i

        Returns
        -------

        """
        gm = self._mat._gm_row
        if isinstance(i, (int, float)) or i.__class__.__name__ in ('int32', 'int64'):
            pass
        else:
            raise Exception(f"can just deal with one dof! now i.__class__ is {i.__class__.__name__}")

        if i < 0:
            i += gm.num_global_dofs
        else:
            pass
        assert i == int(i) and 0 <= i < gm.num_global_dofs, f"i = {i} is wrong."
        i = int(i)
        elements_local_rows = gm.find_rank_locations_of_global_dofs(i)[i]
        for element_local_dof in elements_local_rows:
            rank_element, local_dof = element_local_dof
            data = self._mat[rank_element].tolil()
            data[local_dof, :] = 0
            self._customizations[rank_element] = data.tocsr()

    def identify_row(self, i):
        """identify global row #i: M[i,:] = 0 and M[i, i] = 1, where M means the assembled matrix.

        Parameters
        ----------
        i

        Returns
        -------

        """
        gm = self._mat._gm_row
        if isinstance(i, (int, float)) or i.__class__.__name__ in ('int32', 'int64'):
            pass
        else:
            raise Exception(f"can just deal with one dof! now i.__class__ is {i.__class__.__name__}")

        if i < 0:
            i += gm.num_global_dofs
        else:
            pass

        assert i == int(i) and 0 <= i < gm.num_global_dofs, f"i = {i} is wrong."
        i = int(i)
        elements_local_rows = gm.find_rank_locations_of_global_dofs(i)[i]
        num_global_locations = gm.num_global_locations(i)
        if num_global_locations == 1:
            # this dof only appear at one place, so we just do it on that row!
            if len(elements_local_rows) == 1:  # in the rank where the place is
                rank_element, local_dof = elements_local_rows[0]
                data = self._mat[rank_element].tolil()
                data[local_dof, :] = 0
                data[local_dof, local_dof] = 1
                self._customizations[rank_element] = data.tocsr()
            else:  # in all other ranks, do nothing.
                pass
        else:
            representative_rank, element = gm.find_representative_location(i)
            if RANK == representative_rank:
                representative_local_dof = 1
                for element_local_dof in elements_local_rows:
                    _element, _local_dof = element_local_dof
                    if (_element == element) and representative_local_dof:
                        representative_local_dof = 0
                        data = self._mat[_element].tolil()
                        data[_local_dof, :] = 0
                        data[_local_dof, _local_dof] = 1
                        self._customizations[_element] = data.tocsr()
                    else:
                        data = self._mat[_element].tolil()
                        data[_local_dof, :] = 0
                        self._customizations[_element] = data.tocsr()
            else:
                for element_local_dof in elements_local_rows:
                    _element, _local_dof = element_local_dof
                    data = self._mat[_element].tolil()
                    data[_local_dof, :] = 0
                    self._customizations[_element] = data.tocsr()

    def identify_rows(self, global_dofs):
        """"""
        for i in global_dofs:
            self.identify_row(i)

    def zero_rows(self, global_dofs):
        """"""
        for i in global_dofs:
            self.zero_row(i)


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
                assert A_ij.__class__ is MseHttStaticLocalMatrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"
                row_gm_i = A_ij._gm_row
                col_gm_j = A_ij._gm_col

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

    chain_row_gm = MseHttGatheringMatrix(row_gms)
    chain_col_gm = MseHttGatheringMatrix(col_gms)

    # only adjustments take effect. Customization will be skipped.
    M = _MseHttStaticLocalMatrixBmat(A_2d_list, (row_shape, col_shape))

    return MseHttStaticLocalMatrix(M, chain_row_gm, chain_col_gm, cache_key=M.cache_key)


from scipy.sparse import bmat as sp_bmat


class _MseHttStaticLocalMatrixBmat(Frozen):
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
                    if i in Arc.customize:
                        return 'unique'
                    else:
                        key = Arc._cache_key(i)
                        if key == 'unique':
                            return 'unique'
                        else:
                            keys.append(
                                key
                            )

        return '.'.join(keys)


from numpy import diff
from msehtt.tools.matrix.static.global_ import MseHttGlobalMatrix


___cache_msehtt_assembled_StaticMatrix___ = {}


class MseHttStaticLocalMatrixAssemble(Frozen):
    """"""

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
            if cache in ___cache_msehtt_assembled_StaticMatrix___:
                return ___cache_msehtt_assembled_StaticMatrix___[cache]
            else:
                pass

        else:
            pass

        gm_row = self._M._gm_row
        gm_col = self._M._gm_col

        ROW = list()
        COL = list()
        DAT = list()

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

        if format == 'csc':
            SPA_MATRIX = csc_matrix
        elif format == 'csr':
            SPA_MATRIX = csr_matrix
        else:
            raise Exception

        dep = int(gm_row.num_global_dofs)
        wid = int(gm_col.num_global_dofs)

        A = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))
        A = MseHttGlobalMatrix(A, gm_row, gm_col)
        if isinstance(cache, str):
            ___cache_msehtt_assembled_StaticMatrix___[cache] = A
        else:
            pass
        return A
