# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""
import matplotlib.pyplot as plt
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc, csr_matrix
from tools.frozen import Frozen
from msepy.mesh.elements.main import _DataDictDistributor
from msepy.tools.vector.static.local import MsePyStaticLocalVector
from msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector
import numpy as np

from msepy.tools.matrix.static.assemble import MsePyStaticLocalMatrixAssemble


class MsePyStaticLocalMatrix(Frozen):
    """"""
    def __init__(self, data, gm_row, gm_col, cache_key=None):
        """"""
        if data.__class__ is _DataDictDistributor:
            self._dtype = 'ddd'
            self._data = data  # element-wise csc or csr matrix.
            self._cache_key = data._cache_key_generator

        elif issparse(data) or data == 0:
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

        else:
            raise NotImplementedError(f"MsePyLocalMatrix cannot take data of type {data.__class__}.")

        self._gm0_row = gm_row
        self._gm1_col = gm_col
        self._constant_cache = None
        self._cache = {}
        self._irs = None
        self._customize = _MsePyStaticLocalMatrixCustomize(self)
        self._adjust = _MsePyStaticLocalMatrixAdjust(self)
        self._assemble = MsePyStaticLocalMatrixAssemble(self)
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        shape = self[0].shape
        num_elements = self._gm0_row.num_elements
        self_repr = 'MsePy Static Local Matrix: ' + self._dtype + f"={num_elements}+{shape}"
        return r"<" + self_repr + super_repr

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

    def _get_data_adjusted(self, i):
        """"""
        if i in self.adjust:
            # adjusted data is cached in `adjust.adjustment` anyway!
            data = self.adjust[i]
        else:
            data = self._get_meta_data_from_cache(i)

        assert isspmatrix_csc(data) or isspmatrix_csr(data), f"data for element #i is not sparse."

        return data

    def __getitem__(self, i):
        """Get the final (adjusted and customized) matrix for element #i.
        """
        if i in self.customize:
            data = self.customize[i]
        else:
            data = self._get_data_adjusted(i)

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
        if other.__class__ is MsePyStaticLocalMatrix:

            _matmul = _MatmulMatMat(self, other)
            cache_key = _matmul.cache_key

            static = self.__class__(_matmul, self._gm0_row, other._gm1_col, cache_key=cache_key)

            return static

        elif other.__class__ in (MsePyStaticLocalVector, MsePyRootFormStaticCochainVector):

            _matmul = _mat_mul_mat_vec(self, other)

            assert _matmul is not None, f"for @, we do not accept None data."

            return MsePyStaticLocalVector(_matmul, self._gm0_row)

        else:
            raise NotImplementedError(other)

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

    def _neg_T(self, i):
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
        """data for element #i"""
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
        mat = self._mat._get_data_adjusted(i)
        return self._f * mat

    def _cache_key(self, i):
        """"""
        if i in self._mat.adjust:
            return 'unique'
        else:
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
        elif ck0 == 'constant' and ck1 == 'constant':
            return 'constant'
        else:
            return ck0+'@'+ck1


def _mat_mul_mat_vec(m, v):
    """"""
    vec = v.data  # make sure the 2D data is ready.

    # if isinstance(v, MsePyRootFormStaticCochainVector):
    #     print(v._t)

    if vec is None:
        raise Exception('msepy local vector has no data, cannot @ it.')
    else:
        pass

    if len(m.adjust) == 0:  # we take 2d data (ready).

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

    else:
        raise NotImplementedError(f"take adjusted data.")

    return data


class _MsePyStaticLocalMatrixAdjust(Frozen):
    """"""

    def __init__(self, M):
        """"""
        self._M = M
        self._adjustments = {}  # keys are elements, values are the data.
        self._freeze()

    def __len__(self):
        return len(self._adjustments)

    def __contains__(self, i):
        """Whether element #i is adjusted?"""
        return i in self._adjustments

    def __getitem__(self, i):
        """Return the instance that contains all adjustments of data for element #i."""
        return self._adjustments[i]


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
        """M[i,:] = 0 and M[i, i] = 1

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
