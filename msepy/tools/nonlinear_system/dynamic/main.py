# -*- coding: utf-8 -*-
"""
By Yi Zhang
Created at 5:29 PM on 8/11/2023
"""
from tools.frozen import Frozen
from msepy.tools.linear_system.dynamic.main import MsePyDynamicLinearSystem
from src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem
from msepy.tools.nonlinear_system.dynamic.mda_parser import msepy_mda_parser
from msepy.tools.nonlinear_system.static.local import MsePyStaticLocalNonLinearSystem
from msepy.tools.multidimensional_array.dynamic import MsePyDynamicLocalMDA


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
        all_terms, all_signs = self._nls._n_terms, self._nls._n_signs
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
                parsed_term, text, time_indicator = msepy_mda_parser(term)
                assert parsed_term.__class__ in (MsePyDynamicLocalMDA, ), \
                    f"msepy nonlinear term must be represented by one of ({MsePyDynamicLocalMDA, })"
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
        (
            static_A, static_x, static_b,
            A_texts, x_texts, b_texts,
            A_time_indicating, x_time_indicating, b_time_indicating,
            _str_args,
        ) = self._dls._get_raw_Axb(*args, **kwargs)

        # we pick up the BC, because of nls, for example, the essential bc will take effect later.
        bc = self._dls._bc  # We have to send it to the static local nonlinear system

        parsed_linear_terms = dict()
        parsed_times = dict()

        for i in self._nonlinear_terms:
            parsed_linear_terms[i] = list()
            parsed_times[i] = list()
            for j in range(len(self._nonlinear_terms[i])):
                the_nonlinear_time, times = self._nonlinear_term_call(i, j, *args, **kwargs)
                parsed_linear_terms[i].append(
                    the_nonlinear_time
                )
                parsed_times[i].append(
                    times
                )

        unknowns = list()
        for uk in self.unknowns:
            ati_time = uk.cochain._ati_time_caller(*args, **kwargs)
            unknowns.append(
                uk[ati_time]
            )

        static_nls = MsePyStaticLocalNonLinearSystem(
            # the linear part:
            static_A, static_x, static_b,
            _pr_texts=[A_texts, x_texts, b_texts],
            _time_indicating_text=[A_time_indicating, x_time_indicating, b_time_indicating],
            _str_args=_str_args,
            # the nonlinear part
            bc=bc,
            nonlinear_terms=parsed_linear_terms,
            nonlinear_signs=self._nonlinear_signs,
            nonlinear_texts=self._nonlinear_texts,
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
        real_factor = factor(*args, **kwargs)
        final_term = real_factor * static_local_nonlinear_term
        times = self._nonlinear_time_indicators[i][j](*args, **kwargs)
        return final_term, times
