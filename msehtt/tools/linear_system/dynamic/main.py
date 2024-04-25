# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.wf.mp.linear_system import MatrixProxyLinearSystem
from src.config import _abstract_array_factor_sep, _abstract_array_connector
from src.form.parameters import constant_scalar, ConstantScalar0Form
from src.form.parameters import _factor_parser

_cs1 = constant_scalar(1)

from msehtt.tools.linear_system.dynamic.array_parser import msehtt_root_array_parser


class MseHttDynamicLinearSystem(Frozen):
    """"""

    def __init__(self, wf_mp_ls, base):
        """"""
        assert wf_mp_ls.__class__ is MatrixProxyLinearSystem, f"I need a {MatrixProxyLinearSystem}."
        self._mp_ls = wf_mp_ls
        self._base = base
        self._mp = wf_mp_ls._mp
        self._ls = wf_mp_ls._ls
        self._bc = wf_mp_ls._bc
        self._A = None
        self._x = None
        self._b = None
        self._freeze()

    @property
    def bc(self):
        """Bc"""
        return self._bc

    def apply(self):
        """"""
        self._parse_matrix_block(self._ls.A)
        self._x = self._parse_vector_block(self._ls.x)
        # self._b = self._parse_vector_block(self._ls.b)
        return self

    @property
    def shape(self):
        """Block-wise, I have this shape. So if shape == (3, 3), A of me has 3*3 blocks."""
        return self._mp_ls._ls.A._shape

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
                        DynamicTerm(self, sign, factor, components)
                    )

        #         # noinspection PyTypeChecker
        #         rA[i][j] = DynamicBlockEntry(raw_terms_ij)
        #
        # self._A = rA

    def _parse_vector_block(self, b):
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
                        DynamicTerm(self, sign, factor, components)
                    )

                # # noinspection PyTypeChecker
                # rb[i] = DynamicBlockEntry(raw_terms_i)

        return rb


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
        texts_list = list()
        time_indicating = list()
        for i in self:
            dynamic_term = self[i]
            # below: the three important properties of a dynamic term
            sign, factor, term = dynamic_term.sign, dynamic_term.factor, dynamic_term.component
            factor = factor(*args, **kwargs)
            term = term(*args, **kwargs)

            time_indicating_i = list()
            indicators = dynamic_term.time_indicators
            for indicator in indicators:
                if indicator is None:  # constant term
                    pass
                else:  # indicator must be callable
                    _time = indicator(*args, **kwargs)

                    if isinstance(_time, (int, float)):
                        # this term is determined upon a single time instant.
                        str_time = round(_time, 12)
                        if str_time % 1 == 0:
                            _str = str(int(str_time))
                        else:
                            str_time = round(str_time, 6)
                            _str = str(str_time)
                    elif isinstance(_time, (list, tuple)):
                        _str = list()
                        for _t_ in _time:
                            if _t_ is None:
                                _str.append('None')
                            else:
                                str_time = round(_t_, 12)
                                if str_time % 1 == 0:
                                    _str.append(str(int(str_time)))
                                else:
                                    str_time = round(str_time, 6)
                                    _str.append(str(str_time))
                        _str = r'\left<' + ','.join(_str) + r'\right>'
                    else:
                        raise NotImplementedError()

                    time_indicating_i.append(_str)

            assert isinstance(factor, (int, float)), \
                f"static factor={factor} is wrong, must be a real number."

            # below, we parse the factor into str for printing purpose ------------------------
            _text_factor = round(factor, 12)
            if _text_factor % 1 == 0:
                _text_factor = int(_text_factor)
            else:
                pass

            if _text_factor == 1:
                factor_str = ''
            else:
                if _text_factor == 0:
                    _str_fac = str(0)
                elif (1 / _text_factor) % 1 == 0:
                    _str_fac = r'\frac{1}{' + str(int(1/_text_factor)) + '}'
                else:
                    _str_fac = str(_text_factor)

                factor_str = _str_fac + '*'

            if i == 0 and sign == '+':
                str_sign = ''
            else:
                str_sign = sign

            if hasattr(term, '_gm0_row'):  # must be a MsePy Local matrix
                local_shape = (term._gm0_row.shape[1], term._gm1_col.shape[1])
                texts_list.append(
                    str_sign + factor_str + str(local_shape)
                )
            elif hasattr(term, '_gm'):
                local_shape = term._gm.shape[1]
                texts_list.append(
                    str_sign + factor_str + r'\left[' + str(local_shape) + r'\right]'
                )
            else:
                raise Exception()
            # ===============================================================================

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
            time_indicating.extend(time_indicating_i)

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
        time_indicating = r'+'.join(time_indicating)
        return static, ''.join(texts_list), time_indicating


class DynamicTerm(Frozen):
    """A DynamicTerm is like a term: `- 100 * M @ E`, where sign is `-`, factor is `100`,
    components are `M` and `E`.

    """

    def __init__(self, dls, sign, factor, components):
        assert sign in ('-', '+'), f"sign = {sign} is wrong."
        self._sign = sign

        # parse factor -------------------------------------------------------------------------
        if factor.__class__ is ConstantScalar0Form:
            factor = factor
        elif isinstance(factor, str):
            factor = _factor_parser(factor)
        else:
            raise NotImplementedError(f"cannot parse factor {factor.__class__}: {factor}.")

        assert callable(factor), f"factor must be callable!"
        self._factor = factor  # will be callable, be called to return a particular real number.

        # parse components ---------------------------------------------------------------------
        _components = list()
        _mat_sym_repr = r""
        time_indicators = list()
        for comp_lin_repr in components:

            msehtt_root_array_parser(dls, comp_lin_repr)

            # Mat, sym_repr, time_indicator = msehtt_root_array_parser(dls, comp_lin_repr)

        #     if Mat.__class__ is MsePyStaticLocalMatrix:
        #         Mat = MsePyDynamicLocalMatrix(Mat)
        #     else:
        #         pass
        #
        #     assert Mat.__class__ in (
        #         MsePyDynamicLocalMatrix,
        #         MsePyRootFormDynamicCochainVector,
        #         MsePyDynamicLocalVector,
        #     ), f"{Mat.__class__} cannot be used for RawTerm."
        #
        #     _components.append(Mat)
        #
        #     _mat_sym_repr += sym_repr
        #
        #     assert callable(time_indicator) or time_indicator is None, \
        #         (f"time_indicator {time_indicator} must be callable or None "
        #          f"(means the term does not change over time.)")
        #     time_indicators.append(time_indicator)  # It does not cover time indicator of factor!
        #
        # self._time_indicators = time_indicators  # It does not cover time indicator of factor!
        #
        # # ---- @ all mat together --------------------------------------------------------------
        # if len(_components) == 1:
        #     self._comp = _components[0]
        #
        # else:
        #     self._comp = _components[0] @ _components[1]
        #     for _c in _components[2:]:
        #         self._comp = self._comp @ _c
        #
        # assert self._comp.__class__ in (
        #     MsePyDynamicLocalMatrix,
        #     MsePyRootFormDynamicCochainVector,
        #     MsePyDynamicLocalVector,
        # ), f"{self._comp.__class__} cannot be used for RawTerm."
        #
        # assert self._comp is not None, f"safety check!"
        # self._mat_sym_repr = _mat_sym_repr

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

    @property
    def time_indicators(self):
        """Does not cover time indicator of factor!"""
        return self._time_indicators
