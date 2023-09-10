# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2BasisFunctionWrapper(Frozen):
    """"""

    def __init__(self, bf_qt, csm):
        """"""
        self._bf_qt = bf_qt
        self._csm = csm
        self._freeze()

    def __getitem__(self, index):
        """"""
        if isinstance(index, str):
            bf = self._bf_qt['t']
            if index in self._csm:
                switch_matrix = self._csm[index]
                new_bf = list()
                for csm, bfi in zip(switch_matrix, bf):
                    new_bf.append(
                        csm @ bfi
                    )
                return new_bf
            else:
                return bf
        else:
            bf = self._bf_qt['q']
            if index in self._csm:
                switch_matrix = self._csm[index]
                new_bf = list()
                for csm, bfi in zip(switch_matrix, bf):
                    new_bf.append(
                        csm @ bfi
                    )
                return new_bf
            else:
                return bf
