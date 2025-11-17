# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msepy.tools.linear_system.dynamic.main import MsePyDynamicLinearSystem
from phyem.src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem
from phyem.msepy.tools.nonlinear_system.dynamic.nonlinear_operator_parser import msepy_nonlinear_operator_parser
from phyem.msepy.tools.nonlinear_system.static.local import MsePyStaticLocalNonLinearSystem
from phyem.msepy.tools.nonlinear_operator.dynamic import MsePyDynamicLocalNonlinearOperator


class MsePyDynamicNonLinearSystem(Frozen):
    """We study an abstract `mp_ls` and obtain a raw mse-ls which will future be classified into something else."""

    def __init__(self, mp_nls, msepy_base):
        """"""
        assert mp_nls.__class__ is MatrixProxyNoneLinearSystem, f"I need a {MatrixProxyNoneLinearSystem}!"
        self._mp_nls = mp_nls
        self._nls = mp_nls._nls
        self._dls = MsePyDynamicLinearSystem(mp_nls._mp_ls, msepy_base)  # the dynamic linear system
        self._tfs = None
        self._unknowns = None
        self._nonlinear_factors = dict()
        self._nonlinear_terms = dict()
        self._nonlinear_signs = dict()
        self._nonlinear_time_indicators = dict()
        self._nonlinear_texts = dict()
        self._freeze()

    def apply(self):
        """"""
        self._dls.apply()
        self._parse_nonlinear_terms()
        return self

    @property
    def shape(self):
        return self._dls.shape

    @property
    def bc(self):
        """dynamic linear system directly use bc of the linear system one."""
        return self._dls.bc

    def _pr_temporal_advancing(self, *args, **kwargs):
        """"""
        self._mp_nls._pr_temporal_advancing(*args, **kwargs)

    def _parse_nonlinear_terms(self):
        """"""
        nonlinear_factor = dict()
        parsed_terms = dict()
        parsed_signs = dict()
        time_indicators = dict()
        texts = dict()
        all_terms, all_signs = self._nls._n_terms, self._nls._n_signs  # they are list of list, not dict!
        for i, terms in enumerate(all_terms):
            if len(terms) > 0:
                nonlinear_factor[i] = list()
                parsed_terms[i] = list()   # initialize a list for parsed nonlinear terms for equation #i
                parsed_signs[i] = list()   # initialize a list for parsed signs for equation #i
                time_indicators[i] = list()
                texts[i] = list()
            else:
                pass
            signs = all_signs[i]   # terms, signs of equation #i
            for sign, term in zip(signs, terms):
                nonlinear_factor[i].append(term.factor)
                parsed_term, text, time_indicator = msepy_nonlinear_operator_parser(term)
                assert parsed_term.__class__ in (MsePyDynamicLocalNonlinearOperator,), \
                    f"msepy nonlinear term must be represented by one of ({MsePyDynamicLocalNonlinearOperator, })"
                parsed_terms[i].append(parsed_term)
                parsed_signs[i].append(sign)
                texts[i].append(text)
                time_indicators[i].append(time_indicator)

        self._nonlinear_factors = nonlinear_factor
        self._nonlinear_terms = parsed_terms
        self._nonlinear_signs = parsed_signs
        self._nonlinear_time_indicators = time_indicators
        self._nonlinear_texts = texts

    def __call__(self, *args, **kwargs):
        """"""
        # we first get the variables for the base linear system.
        (
            static_A, static_x, static_b,
            A_texts, x_texts, b_texts,
            A_time_indicating, x_time_indicating, b_time_indicating,
            _str_args,
        ) = self._dls._get_raw_Axb(*args, **kwargs)

        # we pick up the BC, because of nls, for example, the essential bc will take effect later.
        bc = self._dls._bc  # We have to send it to the static local nonlinear system

        # below, we parse the nonlinear terms
        parsed_linear_terms = dict()
        parsed_times = dict()
        parsed_texts = dict()

        for i in self._nonlinear_terms:
            parsed_linear_terms[i] = list()
            parsed_times[i] = list()
            parsed_texts[i] = list()
            for j in range(len(self._nonlinear_terms[i])):
                the_nonlinear_term, times, text = self._nonlinear_term_call(i, j, *args, **kwargs)
                parsed_linear_terms[i].append(
                    the_nonlinear_term  # the factor has been included into this term.
                )
                parsed_times[i].append(
                    times
                )
                parsed_texts[i].append(
                    text
                )

        # --- make static copies of the unknowns ---------
        unknowns = list()
        for uk in self.unknowns:
            ati_time = uk.cochain._ati_time_caller(*args, **kwargs)
            unknowns.append(
                uk[ati_time]
            )

        # make the static nonlinear system ...
        static_nls = MsePyStaticLocalNonLinearSystem(
            # the linear part:
            static_A, static_x, static_b,
            _pr_texts=[A_texts, x_texts, b_texts],
            _time_indicating_text=[A_time_indicating, x_time_indicating, b_time_indicating],
            _str_args=_str_args,
            # the nonlinear part:
            bc=bc,
            nonlinear_terms=parsed_linear_terms,
            nonlinear_signs=self._nonlinear_signs,
            nonlinear_texts=parsed_texts,
            nonlinear_time_indicators=parsed_times,
            test_forms=self.test_forms,
            unknowns=unknowns,
        )

        return static_nls

    @property
    def test_forms(self):
        """the test forms."""
        if self._tfs is None:
            self._tfs = list()
            from msepy.main import base
            msepy_forms = base['forms']
            abstract_tfs = self._mp_nls._mp._wf.test_forms
            for atf in abstract_tfs:
                for mf in msepy_forms:
                    m_f = msepy_forms[mf]
                    if m_f.abstract is atf:
                        self._tfs.append(m_f)
                    else:
                        pass
            assert len(self._tfs) == self.shape[0], f"test forms too many or too less."
        return self._tfs

    @property
    def unknowns(self):
        if self._unknowns is None:
            self._unknowns = list()
            from msepy.main import base
            msepy_forms = base['forms']
            abstract_ufs = self._mp_nls._mp._wf.unknowns
            for auf in abstract_ufs:
                for mf in msepy_forms:
                    m_f = msepy_forms[mf]
                    if m_f.abstract is auf:
                        self._unknowns.append(m_f)
                    else:
                        pass
            assert len(self._unknowns) == self.shape[0], f"test forms too many or too less."
        return self._unknowns

    def _nonlinear_term_call(self, i, j, *args, **kwargs):
        """call the nonlinear term[i][j] (a dynamic object) to make it a static local object."""
        term = self._nonlinear_terms[i][j]
        term.test_form = self.test_forms[i]
        static_local_nonlinear_term = term(*args, **kwargs)
        factor = self._nonlinear_factors[i][j]
        real_number_factor = factor(*args, **kwargs)
        assert isinstance(real_number_factor, (int, float)), f"factor must be parsed as a number now!"
        final_term = real_number_factor * static_local_nonlinear_term
        times = self._nonlinear_time_indicators[i][j](*args, **kwargs)
        text = factor._sym_repr + self._nonlinear_texts[i][j]  # add factor sym to the term text.
        return final_term, times, text

    def _pr_nonlinear_text(self):
        """"""
        nonlinear_text = ''
        for i in range(self.shape[0]):
            if i in self._nonlinear_texts:
                signs = self._nonlinear_signs[i]
                texts = self._nonlinear_texts[i]
                factors = self._nonlinear_factors[i]
                for j, factor in enumerate(factors):
                    sign = signs[j]
                    text = texts[j]
                    if j == 0 and sign == '+':
                        sign = ''
                    else:
                        pass

                    nonlinear_text += sign + factor._sym_repr + text

            else:

                nonlinear_text += r'0'

            if i < self.shape[0] - 1:
                nonlinear_text += r"\\"
            else:
                pass

        nonlinear_text = r"+ \begin{bmatrix}" + nonlinear_text + r"\end{bmatrix}"
        return nonlinear_text

    def pr(self, figsize=(10, 4)):
        """"""

        import matplotlib.pyplot as plt
        import matplotlib
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "DejaVu Sans",
            "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
        })
        matplotlib.use('TkAgg')

        A_text = self._dls._A_pr_text()

        if self._dls._bc is None or len(self._dls._bc) == 0:
            bc_text = ''
        else:
            bc_text = self._dls._bc._bc_text()

        x_text = self._dls._bx_pr_text(self._dls._x)
        b_text = self._dls._bx_pr_text(self._dls._b)

        nonlinear_text = self._pr_nonlinear_text()

        text = A_text + x_text + nonlinear_text + '=' + b_text

        text = r"$" + text + r"$"
        fig = plt.figure(figsize=figsize)
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        plt.text(0.05, 0.5, text + bc_text, ha='left', va='center', size=15)
        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='msepy_DynamicNoneLinearSystem')
        else:
            plt.tight_layout()
            plt.show(block=_setting['block'])

        return fig
