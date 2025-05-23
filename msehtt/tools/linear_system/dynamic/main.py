# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
from src.config import RANK, MASTER_RANK

from tools.frozen import Frozen
from src.wf.mp.linear_system import MatrixProxyLinearSystem
from src.config import _abstract_array_factor_sep, _abstract_array_connector
from src.form.parameters import constant_scalar, ConstantScalar0Form
from src.form.parameters import _factor_parser

from src.wf.mp.linear_system_bc import _EssentialBoundaryCondition

from msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from msehtt.tools.vector.static.local import MseHttStaticLocalVector
from msehtt.static.form.cochain.vector.static import MseHttStaticCochainVector

from msehtt.tools.matrix.dynamic import MseHttDynamicLocalMatrix
from msehtt.tools.vector.dynamic import MseHttDynamicLocalVector
from msehtt.static.form.cochain.vector.dynamic import MseHttDynamicCochainVector

_cs1 = constant_scalar(1)

from msehtt.tools.linear_system.dynamic.array_parser import msehtt_root_array_parser
from msehtt.tools.linear_system.static.local.main import MseHttStaticLocalLinearSystem

from msehtt.tools.linear_system.dynamic.config import MseHttDynamicLinearSystem_Config


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
        self._config = None
        self._freeze()

    @property
    def bc(self):
        """Bc"""
        return self._bc

    def apply(self):
        """"""
        self._parse_matrix_block(self._ls.A)
        self._x = self._parse_vector_block(self._ls.x)
        self._b = self._parse_vector_block(self._ls.b)
        return self

    @property
    def config(self):
        """config for example bc of self."""
        if self._config is None:
            self._config = MseHttDynamicLinearSystem_Config(self)
        return self._config

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

                # noinspection PyTypeChecker
                rA[i][j] = DynamicBlockEntry(raw_terms_ij)

        self._A = rA

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

    def pr(self, figsize=(10, 4)):
        """pr"""
        if RANK != MASTER_RANK:
            return None

        from src.config import _setting, _pr_cache
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "DejaVu Sans",
            "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
        })
        matplotlib.use('TkAgg')

        assert self._A is not None, \
            f"dynamic linear system initialized but not applied, do '.apply()' firstly."

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
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        plt.text(0.05, 0.5, text + bc_text, ha='left', va='center', size=15)
        if _setting['pr_cache']:
            _pr_cache(fig, filename='msepy_DynamicLinearSystem')
        else:
            plt.tight_layout()
            plt.show(block=_setting['block'])

        return fig

    def _pr_temporal_advancing(self, *args, **kwargs):
        """"""
        self._mp_ls._pr_temporal_advancing(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """"""
        (
            static_A, static_x, static_b,
            A_texts, x_texts, b_texts,
            A_time_indicating, x_time_indicating, b_time_indicating,
            _str_args,
         ) = self._get_raw_Axb(*args, **kwargs)

        # static_A, static_x and static_b are used to make a static linear system
        static_linear_system = MseHttStaticLocalLinearSystem(
            static_A, static_x, static_b,
            _pr_texts=[A_texts, x_texts, b_texts],
            _time_indicating_text=[A_time_indicating, x_time_indicating, b_time_indicating],
            _str_args=_str_args,
        )

        # -- below we apply the essential bc ------------------------------------------------------------------------

        if self._bc is None:
            pass
        else:
            for section in self._bc:
                bcs_on_section = self._bc[section]
                for bc in bcs_on_section:
                    if bc.__class__ is _EssentialBoundaryCondition:
                        the_matching_config = None
                        configurations = self.config._configurations
                        for config in configurations[::-1]:
                            if config['type'] == 'essential bc':
                                if config['place'].abstract.manifold._sym_repr == section:
                                    config_root_form = config['root_form']
                                    ith_unknown = bc._i
                                    unknown = static_linear_system.x._x[ith_unknown]
                                    f = unknown._f
                                    if f._is_base():
                                        pass
                                    else:
                                        f = f._base

                                    if config_root_form is f:
                                        the_matching_config = config
                                        break
                                    else:
                                        pass
                                else:
                                    pass
                            else:
                                pass
                        assert the_matching_config is not None, f"configuration not found for essential bc {bc}."
                        # ------------- type ('essential bc', 1) bc -------------------------------------------------
                        if the_matching_config['category'] == 1:
                            place = the_matching_config['place']
                            condition = the_matching_config['condition']
                            ith_unknown = bc._i
                            unknown = static_linear_system.x._x[ith_unknown]
                            time = unknown._time
                            static_linear_system.customize.apply_essential_bc(ith_unknown, place, condition, time)
                        # ===========================================================================================
                        else:
                            raise NotImplementedError()
                    else:
                        pass

        # ----------------- RETURN ----------------------------------------------------------------------------------
        return static_linear_system

    def _get_raw_Axb(self, *args, **kwargs):
        """Get raw Ax=b including natural bc etc."""
        _str_args = ''
        if len(args) == 0 and len(kwargs) == 0:
            pass
        elif len(args) == 0:
            _str_args = str(kwargs)
        elif len(kwargs) == 0:
            _str_args = str(args)
        else:
            _str_args = str(args) + ', ' + str(kwargs)

        num_rows, num_cols = self.shape

        static_A = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        A_texts = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        # do not change '' into something else
        A_time_indicating = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        # do not change '' into something else
        for i in range(num_rows):
            for j in range(num_cols):

                Aij = self._A[i][j]

                if Aij is None:
                    pass

                else:

                    static_Aij, text, time_indicating = Aij(*args, **kwargs)

                    static_A[i][j] = static_Aij

                    A_texts[i][j] = text

                    A_time_indicating[i][j] = time_indicating

        static_x = [None for _ in range(num_cols)]
        x_texts = ['' for _ in range(num_cols)]  # do not change '' into something else
        x_time_indicating = ['' for _ in range(num_cols)]  # do not change '' into something else
        for j in range(num_cols):

            x_j = self._x[j]  # x_j cannot be None

            static_x_j, text, time_indicating = x_j(*args, **kwargs)

            assert static_x_j.__class__ is MseHttStaticCochainVector, \
                f"entry #{j}  of x is not a MsePyRootFormStaticCochainVector!"

            static_x[j] = static_x_j

            x_texts[j] = text

            x_time_indicating[j] = time_indicating

        static_b = [None for _ in range(num_rows)]
        b_texts = ['' for _ in range(num_rows)]            # do not change '' into something else
        b_time_indicating = ['' for _ in range(num_rows)]  # do not change '' into something else
        for i in range(num_rows):

            b_i = self._b[i]

            if b_i is None:
                pass

            else:
                static_b_i, text, time_indicating = b_i(*args, **kwargs)

                static_b[i] = static_b_i

                b_texts[i] = text

                b_time_indicating[i] = time_indicating

        # ----- now we pre-define a static ls to check everything is ok and also to retrieve the gms.
        predefined_sls = MseHttStaticLocalLinearSystem(static_A, static_x, static_b)
        row_gms = predefined_sls._row_gms
        col_gms = predefined_sls._col_gms

        # now we replace all None entries by empty sparse matrices/vectors.
        for i in range(num_rows):
            for j in range(num_cols):
                Aij = static_A[i][j]
                if Aij is None:
                    # noinspection PyTypeChecker
                    static_A[i][j] = MseHttStaticLocalMatrix(
                        0, row_gms[i], col_gms[j], cache_key='zero',
                    )
                else:
                    pass

            if static_b[i] is None:
                # noinspection PyTypeChecker
                static_b[i] = MseHttStaticLocalVector(0, row_gms[i])
            else:
                pass

        return (static_A, static_x, static_b,
                A_texts, x_texts, b_texts,
                A_time_indicating, x_time_indicating, b_time_indicating,
                _str_args)


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

            if hasattr(term, '_gm_row'):  # must be a msehtt local matrix
                # ************** this is not ideal, improve later ***********************************
                if 0 in term._gm_row:
                    shape = (term._gm_row.num_local_dofs(0), term._gm_col.num_local_dofs(0))
                else:
                    shape = ''
                # ***********************************************************************************
                texts_list.append(
                    str_sign + factor_str + str(shape)
                )
            elif hasattr(term, '_gm'):  # must be a msehtt local vector
                # ************** this is not ideal, improve later ***********************************
                if 0 in term._gm:
                    local_shape = term._gm.num_local_dofs(0)
                else:
                    local_shape = ''
                # ***********************************************************************************
                texts_list.append(
                    str_sign + factor_str + r'\left[' + str(local_shape) + r'\right]'
                )
            else:
                raise Exception()
            # =======================================================================================

            if factor == 1:
                if sign == '-':
                    factor_term = - term
                else:
                    factor_term = term

            else:
                if sign == '-':
                    factor_term = (- factor) * term
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
            MseHttStaticLocalMatrix,
            MseHttStaticCochainVector,
            MseHttStaticLocalVector,
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

            Mat, sym_repr, time_indicator = msehtt_root_array_parser(dls, comp_lin_repr)

            if Mat.__class__ is MseHttStaticLocalMatrix:
                Mat = MseHttDynamicLocalMatrix(Mat)
            else:
                pass

            assert Mat.__class__ in (
                MseHttDynamicLocalMatrix,
                MseHttDynamicCochainVector,
                MseHttDynamicLocalVector,
            ), f"{Mat.__class__} cannot be used for RawTerm."

            _components.append(Mat)

            _mat_sym_repr += sym_repr

            assert callable(time_indicator) or time_indicator is None, \
                (f"time_indicator {time_indicator} must be callable or None "
                 f"(means the term does not change over time.)")
            time_indicators.append(time_indicator)  # It does not cover time indicator of factor!

        self._time_indicators = time_indicators  # It does not cover time indicator of factor!

        # ---- @ all mat together --------------------------------------------------------------
        if len(_components) == 1:
            self._comp = _components[0]

        else:
            self._comp = _components[0] @ _components[1]
            for _c in _components[2:]:
                self._comp = self._comp @ _c

        assert self._comp.__class__ in (
            MseHttDynamicLocalMatrix,
            MseHttDynamicCochainVector,
            MseHttDynamicLocalVector,
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

    @property
    def time_indicators(self):
        """Does not cover time indicator of factor!"""
        return self._time_indicators
