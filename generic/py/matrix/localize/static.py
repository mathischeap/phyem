# -*- coding: utf-8 -*-
r"""
"""
from typing import Dict
from tools.frozen import Frozen
from generic.py.gathering_matrix import PyGM
from numpy import diff
from scipy.sparse import csc_matrix, csr_matrix, issparse, isspmatrix_csc, isspmatrix_csr
from generic.py.vector.localize.static import Localize_Static_Vector, Localize_Static_Vector_Cochain

_assembled_Static_Matrix_cache = dict()


class Localize_Static_Matrix(Frozen):
    """"""

    def __init__(self, localized_data, gm_row, gm_col):
        """"""
        assert gm_row.__class__ is PyGM, f'row gm is not a {PyGM}'
        assert gm_col.__class__ is PyGM, f'col gm is not a {PyGM}'
        assert len(gm_row) == len(gm_col), f'gathering matrices length wrong.'
        for index in gm_row:
            assert index in gm_col, f"#index {index} is missing!"

        # --------- when localized_data == 0, we initialize am empty matrix ----------------
        if isinstance(localized_data, (int, float)) and localized_data == 0:
            localized_data = dict()
            for index in gm_row:
                shape = (
                    gm_row.num_local_dofs(index),
                    gm_col.num_local_dofs(index),
                )
                localized_data[index] = csr_matrix(shape)
        else:
            pass
        # =====================================================================================

        if isinstance(localized_data, dict):
            # -------- data check --------------------------------------------------------------
            assert len(localized_data) == len(gm_row) == len(gm_col), f"data length wrong."
            for index in localized_data:
                data_index = localized_data[index]
                assert index in gm_row, f"row gm misses index #{index}"
                assert index in gm_col, f"col gm misses index #{index}"
                assert issparse(data_index), \
                    f"data for index #{index} must be csr or csc matrix, now it is {data_index.__class__}."
                assert data_index.shape == (
                    gm_row.num_local_dofs(index),
                    gm_col.num_local_dofs(index),
                ), f"data shape of element #{index} does not match the gathering matrices."
            # ====================================================================================
            self._meta_data: Dict = localized_data
            self._dtype = 'dict'

        elif callable(localized_data):
            # -------- data check --------------------------------------------------------------
            # ====================================================================================
            self._meta_data = localized_data
            self._dtype = 'realtime'

        else:
            raise Exception(f"cannot accept data={localized_data}.")

        self._gm_row = gm_row
        self._gm_col = gm_col
        self._adjust = _Adjust(self)
        self._customize = _Customize(self)
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Localize_Static_Matrix over {len(self)} elements{super_repr}>"

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
            # noinspection PyCallingNonCallable
            return self._meta_data(index)
        else:
            raise NotImplementedError()

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
        data = self.___get_customize_data___(index)
        assert issparse(data), f"local data must be csc or csr."
        return data

    def __len__(self):
        """How many local elements?"""
        return len(self._gm_row)

    def __iter__(self):
        """iter over all local indices."""
        for index in self._gm_row:
            yield index

    def assemble(self, format='csc', cache=None):
        """assemble self into a `Globalize_Static_Matrix`.

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
            assert isinstance(cache, str), f"'cache' must a string."
            if cache in _assembled_Static_Matrix_cache:
                return _assembled_Static_Matrix_cache[cache]
            else:
                pass

        else:
            pass

        from generic.py.matrix.globalize.static import Globalize_Static_Matrix

        gm_row = self._gm_row
        gm_col = self._gm_col

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

        for i in self:

            Mi = self[i]  # all adjustments and customizations take effect
            indices = Mi.indices
            indptr = Mi.indptr
            data = Mi.data
            nums = diff(indptr)
            row = []
            col = []

            if isspmatrix_csc(Mi):
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    row.extend(gm_row[i][idx])
                    col.extend([gm_col[i][j], ]*num)

            elif isspmatrix_csr(Mi):
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    row.extend([gm_row[i][j], ]*num)
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
        A = Globalize_Static_Matrix(A, gm_row, gm_col)
        if isinstance(cache, str):
            _assembled_Static_Matrix_cache[cache] = A
        return A

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        """"""
        def ___transpose_caller___(index):
            data = self.___get_adjust_data___(index)
            return data.T
        return Localize_Static_Matrix(___transpose_caller___, self._gm_col, self._gm_row)

    def __neg__(self):
        """- self"""
        def ___neg_caller___(index):
            data = self.___get_adjust_data___(index)
            return - data
        return Localize_Static_Matrix(___neg_caller___, self._gm_row, self._gm_col)

    def __matmul__(self, other):
        """ self @ other

        Parameters
        ----------
        other

        Returns
        -------

        """
        if other.__class__ is self.__class__:  # self @ another local static matrix

            assert self._gm_col == other._gm_row, f"A @ B, A._gm_col must == B._gm_row!"

            def ___matmul_mat_caller___(index):
                A = self.___get_adjust_data___(index)
                B = other.___get_adjust_data___(index)
                return A @ B

            return Localize_Static_Matrix(___matmul_mat_caller___, self._gm_row, other._gm_col)

        elif other.__class__ is Localize_Static_Vector or other.__class__ is Localize_Static_Vector_Cochain:

            assert self._gm_col == other._gm, f"A @ b, A._gm_col must == b._gm!"

            def ___matmul_vec_caller___(index):
                A = self.___get_adjust_data___(index)
                b = other.___get_adjust_data___(index)
                return A @ b

            return Localize_Static_Vector(___matmul_vec_caller___, self._gm_row)

        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):

            def ___rmul_float_caller___(index):
                A = self.___get_adjust_data___(index)
                return other * A

            return Localize_Static_Matrix(___rmul_float_caller___, self._gm_row, self._gm_col)

        else:
            raise NotImplementedError()

    def __add__(self, other):
        """self + other"""

        if other.__class__ is self.__class__:

            assert self._gm_col == other._gm_col
            assert self._gm_row == other._gm_row

            def ___add_caller___(index):
                A = self.___get_adjust_data___(index)
                B = other.___get_adjust_data___(index)
                return A + B

            return Localize_Static_Matrix(___add_caller___, self._gm_row, self._gm_col)

        else:
            raise NotImplementedError()

    def __sub__(self, other):
        """self - other"""

        if other.__class__ is self.__class__:

            assert self._gm_col == other._gm_col
            assert self._gm_row == other._gm_row

            def ___sub_caller___(index):
                A = self.___get_adjust_data___(index)
                B = other.___get_adjust_data___(index)
                return A - B

            return Localize_Static_Matrix(___sub_caller___, self._gm_row, self._gm_col)

        else:
            raise NotImplementedError()


class _Adjust(Frozen):
    """"""

    def __init__(self, M):
        """"""
        self._M = M
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

    def __init__(self, M):
        """"""
        self._M = M
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

    def identify_row(self, i):
        """identify global row #i: M[i,:] = 0 and M[i, i] = 1, where M means the assembled matrix.

        Parameters
        ----------
        i

        Returns
        -------

        """
        elements_local_rows = self._M._gm_row._find_elements_and_local_indices_of_dofs(i)

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

        elements_local_rows = self._M._gm_row._find_elements_and_local_indices_of_dofs(global_dofs)

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
        elements_local_rows = self._M._gm_row._find_elements_and_local_indices_of_dofs(global_dofs)
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
