# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from src.config import _abstract_array_factor_sep, _abstract_array_connector
from src.form.parameters import constant_scalar, ConstantScalar0Form, _constant_scalar_parser
from msepy.tools.linear_system.array_parser import msepy_root_array_parser
_cs1 = constant_scalar(1)


class MsePyRawLinearSystem(Frozen):
    """We study an abstract `mp_ls` and obtain a raw mse-ls which will future be classified into something else."""

    def __init__(self, mp_ls):
        """"""
        A = self._parse_matrix_block(mp_ls._ls.A)
        x = self._parse_vector_block(mp_ls._ls.x)
        b = self._parse_vector_block(mp_ls._ls.b)
        self._bc = mp_ls._bc
        self._freeze()

    @property
    def bc(self):
        return self._bc

    def _parse_matrix_block(self, A):
        """"""
        Si, Sj = A._shape
        rA = [[None for _ in range(Sj)] for _ in range(Si)]  # A for this raw ls.
        for i, j in A:
            Aij = A(i, j)  # the block at A[i][j]

            raw_terms_ij = list()
            for aij, sign in zip(*Aij):

                # this block may have multiple terms, we go through these terms (and their signs)
                if _abstract_array_factor_sep in aij._lin_repr:  # there is a factor for this term
                    factor, components = aij._lin_repr.split(_abstract_array_factor_sep)
                else:
                    factor = _cs1
                    components = aij._lin_repr

                components = components.split(_abstract_array_connector)

                raw_terms_ij.append(
                    RawTerm(sign, factor, components)
                )

            # noinspection PyTypeChecker
            rA[i][j] = RawBlock(raw_terms_ij)


    def _parse_vector_block(self, b):
        """"""
        for i in b:
            bi = b(i)


class RawBlock(Frozen):
    """"""

    def __init__(self, raw_terms):
        self._raw_terms = raw_terms
        self._freeze()


class RawTerm(Frozen):
    """"""

    def __init__(self, sign, factor, components):
        assert sign in ('-', '+'), f"sign = {sign} is wrong."
        self._sign = sign

        # parse factor -----------
        if factor.__class__ is ConstantScalar0Form:
            factor = _constant_scalar_parser(factor)
        else:
            raise NotImplementedError()

        self._factor = factor

        # parse components -----------
        for comp_lin_repr in components:

            MorV = msepy_root_array_parser(comp_lin_repr)

        self._freeze()


if __name__ == '__main__':
    # python 
    pass
