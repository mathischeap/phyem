# -*- coding: utf-8 -*-
r"""
"""
from importlib import import_module

from phyem.tools.frozen import Frozen


class MseHtt_SpaceLambda_Wedge_Matrix(Frozen):
    r""""""

    def __init__(self, space0, space1):
        r""""""
        assert space0.tpm is space1.tpm
        self._space0 = space0
        self._space1 = space1
        self._freeze()

    def __call__(self, degree0, degree1):
        r""""""
        m0 = self._space0.m
        n0 = self._space0.n
        k0 = self._space0.abstract.k
        m1 = self._space1.m
        n1 = self._space1.n
        k1 = self._space1.abstract.k
        orientation0 = self._space0.orientation
        orientation1 = self._space1.orientation
        indicator0 = f"m{m0}n{n0}k{k0}"
        indicator1 = f"m{m1}n{n1}k{k1}"
        path = self.__repr__().split('main.')[0][1:] + f"WM_{indicator0}"
        module = import_module(path)
        method_name = 'wedge_matrix_Lambda__' + indicator0 + '_w_' + indicator1
        method_name_orientation = ('wedge_matrix_Lambda__' +
                                   indicator0 + '_' + orientation0 + '_w_' + indicator1 + '_' + orientation1)
        if hasattr(module, method_name):
            return getattr(module, method_name)(self._space0.tpm, degree0, degree1)
        elif hasattr(module, method_name_orientation):
            return getattr(module, method_name_orientation)(self._space0.tpm, degree0, degree1)
        else:
            raise NotImplementedError(method_name, method_name_orientation)
