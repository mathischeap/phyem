# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
import numpy as np

from tools.frozen import Frozen


class MsePyStaticLocalMDA(Frozen):
    """"""

    def __init__(self, local_static_MDA, particular_correspondence):
        """

        Parameters
        ----------
        local_static_MDA
        particular_correspondence:
            the corresponding msepy forms at particular time instances.
        """
        if isinstance(local_static_MDA, dict) and \
           all([isinstance(local_static_MDA[_], np.ndarray) for _ in local_static_MDA]):
            # local_static_MDA is provided as a dict of ndarray(s).
            self._dtype = 'ndarray'
            self._data = local_static_MDA
        elif callable(local_static_MDA):  # local_static_MDA(e) gives data of element #e
            self._dtype = 'realtime'
            self._data = local_static_MDA
        else:
            raise NotImplementedError()
        self._correspondence = particular_correspondence

        self._freeze()

    def __getitem__(self, e):
        """"""
        if self._dtype == 'realtime':
            return self._data(e)
        elif self._dtype == 'ndarray':
            # noinspection PyUnresolvedReferences
            return self._data[e]
        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            # c * self; c is a number

            helper = _RmulFactorHelper(self._data, other)
            # noinspection PyTypeChecker
            return self.__class__(helper, self._correspondence)

        else:
            raise Exception()


class _RmulFactorHelper(Frozen):
    """"""
    def __init__(self, data, factor):
        """"""
        self._d = data
        self._f = factor
        self._freeze()

    def __call__(self, e):
        """data of element #e."""
        return self._f * self._d[e]
