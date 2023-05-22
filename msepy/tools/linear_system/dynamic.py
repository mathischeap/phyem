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
from msepy.tools.matrix.dynamic import MsePyDynamicLocalMatrixVector
from msepy.tools.vector.dynamic import MsePyDynamicLocalVector
from msepy.form.cochain.vector.dynamic import MsePyRootFormDynamicCochainVector
from src.config import _abstract_array_factor_sep, _abstract_array_connector
from src.form.parameters import constant_scalar, ConstantScalar0Form
from src.form.parameters import _factor_parser
from msepy.tools.linear_system.array_parser import msepy_root_array_parser

from msepy.tools.linear_system.static.main import MsePyStaticLinearSystem

from msepy.tools.vector.static.local import MsePyStaticLocalVector
from msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector

_cs1 = constant_scalar(1)


class MsePyDynamicLinearSystem(Frozen):
    """We study an abstract `mp_ls` and obtain a raw mse-ls which will future be classified into something else."""

    def __init__(self, mp_ls):
        """"""
        self._mp_ls = mp_ls
        self.bc = mp_ls._bc
        self._A = None
        self._x = None
        self._b = None
        self._freeze()

    @property
    def bc(self):
        return self._bc

    @bc.setter
    def bc(self, bc):
        if bc is None:
            self._bc = None
        else:
            pass
            # raise NotImplementedError(f"We will parse the bc to msepy bc.")

    @property
    def shape(self):
        return self._mp_ls._ls.A._shape

    def apply(self):
        """"""
        self._parse_matrix_block(self._mp_ls._ls.A)
        self._x = self._parse_vector_block(self._mp_ls._ls.x)
        self._b = self._parse_vector_block(self._mp_ls._ls.b)
        return self

    def __call__(self, *args, **kwargs):
        """"""
        num_rows, num_cols = self.shape

        static_A = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        for i in range(num_rows):
            for j in range(num_cols):

                Aij = self._A[i][j]

                if Aij is None:
                    pass
                else:

                    static_Aij = Aij(*args, **kwargs)

                    static_A[i][j] = static_Aij

        static_x = [None for _ in range(num_cols)]
        for j in range(num_cols):

            x_j = self._x[j]  # x_j cannot be None

            static_x_j = x_j(*args, **kwargs)

            assert static_x_j.__class__ is MsePyRootFormStaticCochainVector, \
                f"entry #{j}  of x is not a MsePyRootFormStaticCochainVector!"
            static_x[j] = static_x_j

        static_b = [None for _ in range(num_rows)]
        for i in range(num_rows):

            b_i = self._b[i]

            if b_i is None:
                pass
            else:
                static_b_i = b_i(*args, **kwargs)

                static_b[i] = static_b_i

        # probably, we will need to first parse the bc here to make it static and then pass it to static ls.

        return MsePyStaticLinearSystem(static_A, static_x, static_b, bc=self._bc)

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
                        DynamicTerm(sign, factor, components)
                    )

                # noinspection PyTypeChecker
                rA[i][j] = DynamicBlockEntry(raw_terms_ij)

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
                        DynamicTerm(sign, factor, components)
                    )

                # noinspection PyTypeChecker
                rb[i] = DynamicBlockEntry(raw_terms_i)

        return rb

    def _A_pr_text(self):
        """_A_pr_text"""
        A_text = r""
        I_, J_ = self.shape

        for i in range(I_):
            for j in range(J_):
                Aij = self._A[i][j]
                if Aij is None:
                    A_text += '0'
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

    @staticmethod
    def _bx_pr_text(b_or_x):
        """"""
        text = r""

        I_ = len(b_or_x)

        for i in range(I_):
            bi = b_or_x[i]

            if bi is None:
                text += '0'
            else:
                text += bi._pr_text()

            if i < I_ - 1:
                text += r"\\"

        text = r"\begin{bmatrix}" + text + r"\end{bmatrix}"

        return text

    def pr(self, figsize=(10, 6)):
        """pr"""
        A_text = self._A_pr_text()

        if self._bc is None or len(self._bc) == 0:
            bc_text = ''
        else:
            bc_text = self._bc._bc_text()

        x_text = self._bx_pr_text(self._x)
        b_text = self._bx_pr_text(self._b)

        text = A_text + x_text + '=' + b_text
        text = r"$" + text + r"$"
        fig = plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        plt.text(0.05, 0.5, text + bc_text, ha='left', va='center', size=15)
        plt.tight_layout()
        from src.config import _matplot_setting
        plt.show(block=_matplot_setting['block'])

        return fig


class DynamicBlockEntry(Frozen):
    """A bunch of dynamic terms."""

    def __init__(self, raw_terms):
        for rt in raw_terms:
            assert rt.__class__ is DynamicTerm
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

    def __call__(self, *args, **kwargs):
        """"""
        factor_terms = list()
        for i in self:
            dynamic_term = self[i]
            sign, factor, term = dynamic_term.sign, dynamic_term.factor, dynamic_term.component

            factor = factor(*args, **kwargs)
            term = term(*args, **kwargs)

            assert isinstance(factor, (int, float)), f"static factor={factor} is wrong, must be a real number."

            if factor == 1:
                if sign == '-':
                    factor_term = - term
                else:
                    factor_term = term

            else:

                if sign == '-':
                    factor_term = - factor * term
                else:
                    factor_term = factor * term

            factor_terms.append(factor_term)

        static = factor_terms[0]
        if len(factor_terms) == 1:
            pass
        else:
            for ft in factor_terms[1:]:
                static += ft

        assert static.__class__ in (
            MsePyStaticLocalMatrix,
            MsePyRootFormStaticCochainVector,
            MsePyStaticLocalVector,
        ), f"static={static} is wrong."

        return static


class DynamicTerm(Frozen):
    """"""

    def __init__(self, sign, factor, components):
        assert sign in ('-', '+'), f"sign = {sign} is wrong."
        self._sign = sign

        # parse factor -----------
        if factor.__class__ is ConstantScalar0Form:
            factor = factor
        elif isinstance(factor, str):
            factor = _factor_parser(factor)
        else:
            raise NotImplementedError(f"cannot parse factor {factor.__class__}: {factor}.")

        assert callable(factor), f"factor must be callable!"
        self._factor = factor  # will be callable, be called to return a particular real number.

        # parse components -----------
        _components = list()
        _mat_sym_repr = r""
        for comp_lin_repr in components:

            Mat, sym_repr = msepy_root_array_parser(comp_lin_repr)

            if Mat.__class__ is MsePyStaticLocalMatrix:
                Mat = MsePyDynamicLocalMatrixVector(Mat)
            else:
                pass

            assert Mat.__class__ in (
                MsePyDynamicLocalMatrixVector,
                MsePyRootFormDynamicCochainVector,
                MsePyDynamicLocalVector,
            ), f"{Mat.__class__} cannot be used for RawTerm."

            _components.append(Mat)

            _mat_sym_repr += sym_repr

        # ---- @ all mat together -------------------
        if len(_components) == 1:
            self._comp = _components[0]

        else:
            self._comp = _components[0] @ _components[1]
            for _c in _components[2:]:
                self._comp = self._comp @ _c

        assert self._comp.__class__ in (
            MsePyDynamicLocalMatrixVector,
            MsePyRootFormDynamicCochainVector,
            MsePyDynamicLocalVector,
        ), f"{self._comp.__class__} cannot be used for RawTerm."

        assert self._comp is not None, f"safety check!"
        self._mat_sym_repr = _mat_sym_repr

        self._freeze()

    def _pr_text(self):
        """_pr_text"""
        # put sign no matter it is + or -.
        # noinspection PyUnresolvedReferences
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
    def component(self):
        """components"""
        return self._comp
