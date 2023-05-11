# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""
from scipy.sparse import issparse

from tools.frozen import Frozen
from msepy.mesh.elements import _DataDictDistributor


class MsePyStaticLocalMatrix(Frozen):
    """"""
    def __init__(self, data, gm_row, gm_col, cache_key=None):
        """"""
        if data.__class__ is _DataDictDistributor:
            self._dtype = 'ddd'
            self._data = data  # csc or csr matrix.
            self._cache_key = data._cache_key_generator
        elif issparse(data):
            self._dtype = 'constant'
            self._data = data
            shape0, shape1 = data.shape  # must be regular gathering matrix, so `.shape` does not raise Error.
            assert shape0 == gm_row.shape[1], f"row shape wrong"
            assert shape1 == gm_col.shape[1], f"col shape wrong"
            self._cache_key = self._constant_cache_key
        elif callable(data):
            self._dtype = 'realtime'
            self._data = data
            assert cache_key is not None, f"when provided callable data, must provide cache_key."
            self._cache_key = cache_key
        else:
            raise NotImplementedError(f"MsePyLocalMatrix cannot take data of type {data.__class__}.")
        self._gm0_row = gm_row
        self._gm1_col = gm_col
        self._constant_cache = None
        self._cache = {}
        self._customize = _MsePyStaticLocalMatrixCustomize(self)
        self._adjust = _MsePyStaticLocalMatrixAdjust(self)
        self._freeze()

    def _constant_cache_key(self, i):
        """"""
        assert i in self, f"i={i} is out of range."
        return 'constant'

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

        elif ck == 'unique':  # we do not cache at all. Use the meta-data.
            return self.___get_meta_data___(i)

        else:  # otherwise, we do dynamic caching.
            if ck in self._cache:
                data = self._cache[ck]
            else:
                data = self.___get_meta_data___(i)
                self._cache[ck] = data
            return data

    def _get_adjusted_data(self, i):
        """"""
        data = self._get_meta_data_from_cache(i)

        assert issparse(data), f"data for element #i is not sparse."
        if i in self.adjust:
            # get (__getitem__) the correct _ElementCustomization and let it take effect (__call__).
            return self.adjust[i](data)
        else:
            return data

    def __getitem__(self, i):
        """Get the final (adjusted and customized) matrix for element #i.
        """
        data = self._get_adjusted_data(i)

        assert issparse(data), f"data for element #i is not sparse."
        if i in self.customize:
            # get (__getitem__) the correct _ElementCustomization and let it take effect (__call__).
            return self.customize[i](data)
        else:
            return data

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

    @property
    def customize(self):
        """Will not touch the data. Modification will be applied addition to the data."""
        return self._customize

    @property
    def adjust(self):
        """Will not touch the data. Modification will be applied addition to the data."""
        return self._adjust

    def adjust(self):
        """Will touch the data (by making new copies, the meta-data will not be touched).
        Modification will be applied to the data."""
        if self._adjust is None:
            self._adjust = _MsePyStaticLocalMatrixAdjust(self)
        return self._adjust

    @staticmethod
    def is_static():
        """static"""
        return True

    @property
    def T(self):
        """The `customization` will not have an effect. The adjustment will be taken into account."""
        return MsePyStaticLocalMatrix(
            self._data_T,
            self._gm1_col,
            self._gm0_row,
            cache_key=self._cache_key,
        )

    def _data_T(self, i):
        """"""
        data = self._get_adjusted_data(i)
        return data.T

    def __matmul__(self, other):
        """self @ other.

        The `customization` of both entries will not have an effect. The adjustments will be taken into account.
        """
        if other.__class__ is MsePyStaticLocalMatrix:

            _matmul = _Matmul(self, other)
            cache_key = _matmul.cache_key

            return MsePyStaticLocalMatrix(_matmul, self._gm0_row, other._gm1_col, cache_key=cache_key)

        else:
            raise NotImplementedError()


class _Matmul(Frozen):
    """"""

    def __init__(self, M0, M1):
        """"""
        self._m0 = M0
        self._m1 = M1
        self._freeze()

    def __call__(self, i):
        """"""
        data0 = self._m0._get_adjusted_data()
        data1 = self._m1._get_adjusted_data()
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


class _MsePyStaticLocalMatrixCustomize(Frozen):
    """"""

    def __init__(self, M):
        """"""
        self._M = M
        self._customization = {}
        self._freeze()

    def __contains__(self, i):
        """Whether element #i is customized?"""
        return i in self._customization

    def __getitem__(self, i):
        """Return the instance that contains all customization of data for element #i."""
        return self._customization[i]


class _ElementCustomization(Frozen):
    """"""

    def __init__(self, i):
        """"""
        self._i = i
        self._freeze()

    def __call__(self, data):
        """Let the customization take effect."""
        # todo
        return data


class _MsePyStaticLocalMatrixAdjust(Frozen):
    """"""

    def __init__(self, M):
        """"""
        self._M = M
        self._adjustment = {}
        self._freeze()

    def __contains__(self, i):
        """Whether element #i is adjusted?"""
        return i in self._adjustment

    def __getitem__(self, i):
        """Return the instance that contains all adjustments of data for element #i."""
        return self._adjustment[i]


class _ElementAdjustment(Frozen):
    """"""

    def __init__(self, i):
        """"""
        self._i = i
        self._freeze()

    def __call__(self, data):
        """Let the adjustments take effect."""
        # todo
        return data
