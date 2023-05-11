# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')
from tools.frozen import Frozen
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.tools.matrix.dynamic.local import MsePyDynamicLocalMatrix
from src.config import _abstract_array_factor_sep, _abstract_array_connector
from src.form.parameters import constant_scalar, ConstantScalar0Form, _constant_scalar_parser
from msepy.tools.linear_system.array_parser import msepy_root_array_parser
_cs1 = constant_scalar(1)


class MsePyRawLinearSystem(Frozen):
    """We study an abstract `mp_ls` and obtain a raw mse-ls which will future be classified into something else."""

    def __init__(self, mp_ls):
        """"""
        self._mp_ls = mp_ls
        self._bc = mp_ls._bc
        self._A = None
        self._x = None
        self._b = None
        self._freeze()

    @property
    def bc(self):
        return self._bc

    @property
    def shape(self):
        return self._mp_ls._ls.A._shape

    def __call__(self, *args, **kwargs):
        """"""
        self._parse_matrix_block(self._mp_ls._ls.A)
        # self._x = self._parse_vector_block(self._mp_ls._ls.x)
        self._b = self._parse_vector_block(self._mp_ls._ls.b)
        return self

    def _parse_matrix_block(self, A):
        """"""
        Si, Sj = A._shape
        rA = [[None for _ in range(Sj)] for _ in range(Si)]  # A for this raw ls.
        for i, j in A:
            Aij = A(i, j)  # the block at A[i][j]
            if Aij == ([], []):
                pass
            else:
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
                rA[i][j] = RawBlockEntry(raw_terms_ij)
        self._A = rA

    @staticmethod
    def _parse_vector_block(b):
        """"""
        s = b._shape
        rb = [None for _ in range(s)]  # raw b
        for i in range(s):
            bi = b(i)
            if bi == ([], []):
                pass
            else:
                raw_terms_i = list()
                for bi_, sign in zip(*bi):
                    # this block may have multiple terms, we go through these terms (and their signs)
                    if _abstract_array_factor_sep in bi_._lin_repr:  # there is a factor for this term
                        factor, components = bi_._lin_repr.split(_abstract_array_factor_sep)
                    else:
                        factor = _cs1
                        components = bi_._lin_repr

                    components = components.split(_abstract_array_connector)

                    raw_terms_i.append(
                        RawTerm(sign, factor, components)
                    )

                # noinspection PyTypeChecker
                rb[i] = RawBlockEntry(raw_terms_i)

        return rb

    def _A_pr_text(self):
        """_A_pr_text"""
        A_text = r""
        I_, J_ = self.shape
        for i in range(I_):
            for j in range(J_):
                Aij = self._A[i][j]
                if Aij is None:
                    pass
                else:
                    A_text += Aij._pr_text()

                if j < J_ - 1:
                    A_text += '&'
                else:
                    pass
            if i < I_ - 1:
                A_text += r"\\"
        A_text = r"\begin{bmatrix}" + A_text + r"\end{bmatrix}"

        return A_text

    def pr(self, figsize=(10, 6)):
        """pr"""
        A_text = self._A_pr_text()

        if self._bc is None or len(self._bc) == 0:
            bc_text = ''
        else:
            bc_text = self._bc._bc_text()

        A_text = r"$" + A_text + r"$"
        fig = plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        plt.text(0.05, 0.5, A_text + bc_text, ha='left', va='center', size=15)
        plt.tight_layout()
        from src.config import _matplot_setting
        plt.show(block=_matplot_setting['block'])

        return fig


class RawBlockEntry(Frozen):
    """A bunch of RawTerms"""

    def __init__(self, raw_terms):
        self._raw_terms = raw_terms  # a list of RawTerm
        self._freeze()

    def __len__(self):
        """how many RawTerm I have?"""
        return len(self._raw_terms)

    def __iter__(self):
        """go through all indices of raw terms."""
        for t in range(len(self)):
            yield t

    def __getitem__(self, t):
        return self._raw_terms[t]

    def _pr_text(self):
        pr_text = r''
        for i in self:
            term_text = self[i]._pr_text()
            if i == 0 and term_text[0] == '+':
                pr_text += term_text[1:]
            else:
                pr_text += term_text
        return pr_text


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

        self._factor = factor  # will be callable, be called to return a particular real number.

        # parse components -----------
        _components = list()
        _mat_sym_repr = r""
        for comp_lin_repr in components:

            Mat, sym_repr = msepy_root_array_parser(comp_lin_repr)

            if Mat.__class__ is MsePyStaticLocalMatrix:
                Mat = MsePyDynamicLocalMatrix(Mat)
            elif Mat.__class__ is MsePyDynamicLocalMatrix:
                pass
            else:
                raise NotImplementedError()

            _components.append(Mat)

            _mat_sym_repr += sym_repr

        # ---- @ all mat together -------------------
        if len(_components) == 1:
            self._mat = _components
        else:
            self._mat = _components[0] @ _components[1]
            for _c in _components[2:]:
                self._mat = self._mat @ _c

        #
        self._mat_sym_repr = _mat_sym_repr

        self._freeze()

    def _pr_text(self):
        """_pr_text"""
        # put sign no matter it is + or -.
        return self.sign + self._factor._pr_text() + self._mat_sym_repr

    @property
    def sign(self):
        """sign"""
        return self._sign

    @property
    def factor(self):
        """factor"""
        return self._factor

    @property
    def mat(self):
        """components"""
        return self._mat


if __name__ == '__main__':
    # python 
    pass
