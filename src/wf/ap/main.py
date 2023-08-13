# -*- coding: utf-8 -*-
"""
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


class AlgebraicProxy(Frozen):
    """"""

    def __init__(self, wf):
        self._fully_resolved = True
        self.___all_linear___ = True
        self._parse_terms(wf)
        self._parse_unknowns_test_vectors(wf)
        self._wf = wf
        self._bc = wf._bc
        self._evs = None
        self._freeze()

    def _parse_terms(self, wf):
        """"""
        wf_td = wf._term_dict
        wf_sd = wf._sign_dict
        term_dict = dict()   # the terms for the AP equation
        sign_dict = dict()   # the signs for the AP equation
        color_dict = dict()
        ind_dict = dict()
        indexing = dict()
        linearity_dict = dict()
        for i in wf_td:
            term_dict[i] = ([], [])
            sign_dict[i] = ([], [])
            color_dict[i] = ([], [])
            ind_dict[i] = ([], [])
            linearity_dict[i] = ([], [])
            k = 0
            for j, terms in enumerate(wf_td[i]):
                for m, term in enumerate(terms):
                    old_sign = wf_sd[i][j][m]
                    try:
                        ap, new_sign, linearity = term.ap(test_form=wf.test_forms[i])
                        new_sign = self._parse_sign(new_sign, old_sign)
                        color = 'k'
                        assert linearity in ('linear', 'nonlinear'), \
                            f"`linearity` must be among ('linear', 'nonlinear'), now it is {linearity}."
                        if self.___all_linear___ and linearity == 'nonlinear':
                            self.___all_linear___ = False
                        else:
                            pass

                    except NotImplementedError:

                        ap = term
                        new_sign = old_sign
                        self._fully_resolved = False
                        color = 'r'
                        linearity = 'unknown'
                        self.___all_linear___ = False

                    index = str(i) + '-' + str(k)
                    k += 1
                    indexing[index] = (new_sign, ap)
                    ind_dict[i][j].append(index)
                    term_dict[i][j].append(ap)
                    color_dict[i][j].append(color)
                    sign_dict[i][j].append(new_sign)
                    linearity_dict[i][j].append(linearity)

        self._term_dict = term_dict
        self._sign_dict = sign_dict
        self._color_dict = color_dict
        self._indexing = indexing
        self._ind_dict = ind_dict
        self._linearity_dict = linearity_dict

    @property
    def linearity(self):
        """The system is linear, nonlinear or unknown?"""
        if self._fully_resolved:
            if self.___all_linear___:
                return 'linear'
            else:
                return 'nonlinear'
        else:
            return 'unknown'

    @staticmethod
    def _parse_sign(s0, s1):
        """parse sign"""
        return '+' if s0 == s1 else '-'

    def _parse_unknowns_test_vectors(self, wf):
        """parse unknowns test vectors."""
        assert wf.unknowns is not None, f"pls first set unknowns of the weak formulation."
        assert wf.test_forms is not None, f"Weak formulation should have specify test forms."

        self._unknowns = list()
        for wfu in wf.unknowns:
            self._unknowns.append(wfu._ap)

        self._tvs = list()
        for tf in wf.test_forms:
            self._tvs.append(tf._ap)

    @property
    def unknowns(self):
        """unknowns"""
        return self._unknowns

    @property
    def test_vectors(self):
        """test vectors."""
        return self._tvs

    @property
    def elementary_vectors(self):
        """elementary vectors."""
        if self._evs is None:
            self._evs = list()
            for ef in self._wf.elementary_forms:
                self._evs.append(ef.ap())
            self._evs = tuple(self._evs)
        return self._evs

    def pr(self, indexing=True):
        """Print the representations"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            pass
        seek_text = self._wf._mesh.manifold._manifold_text()
        seek_text += r'seek $\left('
        form_sr_list = list()
        space_sr_list = list()
        for un in self.unknowns:
            form_sr_list.append(rf' {un._sym_repr}')
            space_sr_list.append(rf"{un._shape_text()}")
        seek_text += ','.join(form_sr_list)
        seek_text += r'\right) \in '
        seek_text += r'\times '.join(space_sr_list)
        seek_text += '$, such that\n'
        symbolic = ''
        number_equations = len(self._term_dict)
        for i in self._term_dict:
            for t, terms in enumerate(self._term_dict[i]):
                if len(terms) == 0:
                    symbolic += '0'
                else:

                    for j, term in enumerate(terms):
                        sign = self._sign_dict[i][t][j]
                        term = self._term_dict[i][t][j]
                        color = self._color_dict[i][t][j]

                        term_sym_repr = term._sym_repr

                        if indexing:
                            index = self._ind_dict[i][t][j].replace('-', r'\text{-}')
                            term_sym_repr = r'\underbrace{' + term_sym_repr + r'}_{' + \
                                rf"{index}" + '}'
                        else:
                            pass

                        if color == 'r':  # this term is not converted into algebraic proxy yet!
                            term_sym_repr = r"\left[" + term_sym_repr + r"\right]^{!}"
                        else:
                            pass

                        if j == 0:
                            if sign == '+':
                                symbolic += term_sym_repr
                            elif sign == '-':
                                symbolic += '-' + term_sym_repr
                            else:
                                raise Exception()
                        else:
                            symbolic += ' ' + sign + ' ' + term_sym_repr

                if t == 0:
                    symbolic += ' &= '

            symbolic += r'\quad &&\forall ' + \
                        self.test_vectors[i]._sym_repr + \
                        r'\in' + \
                        self.test_vectors[i]._shape_text()

            if i < number_equations - 1:
                symbolic += r',\\'
            else:
                symbolic += '.'

        symbolic = r"$\left\lbrace\begin{aligned}" + symbolic + r"\end{aligned}\right.$"
        if self._bc is None or len(self._bc) == 0:
            bc_text = ''
        else:
            bc_text = self._bc._bc_text()

        if indexing:
            figsize = (12, 3 * len(self._term_dict))
        else:
            figsize = (12, 3 * len(self._term_dict))

        fig = plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        plt.text(0.05, 0.5, seek_text + '\n' + symbolic + bc_text, ha='left', va='center', size=15)
        plt.tight_layout()
        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='algebraicProxy')
        else:
            plt.show(block=_setting['block'])
        return fig


if __name__ == '__main__':
    # python src/wf/ap/rct.py
    import __init__ as ph

    samples = ph.samples

    oph = samples.pde_canonical_pH(n=3, p=3)[0]
    a3, b2 = oph.unknowns
    # oph.pr()

    a31 = a3 @ 1

    # wf = oph.test_with(oph.unknowns, sym_repr=[r'v^3', r'u^2'])
    # wf = wf.derive.integration_by_parts('1-1')
    # # wf.pr(indexing=True)
    #
    # td = wf.td
    # td.set_time_sequence()  # initialize a time sequence
    #
    # td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
    # td.differentiate('0-0', 'k-1', 'k')
    # td.average('0-1', b2, ['k-1', 'k'])
    #
    # td.differentiate('1-0', 'k-1', 'k')
    # td.average('1-1', a3, ['k-1', 'k'])
    # td.average('1-2', a3, ['k-1/2'])
    # dt = td.time_sequence.make_time_interval('k-1', 'k')
    #
    # wf = td()
    #
    # # wf.pr()
    #
    # wf.unknowns = [
    #     a3 @ td.time_sequence['k'],
    #     b2 @ td.time_sequence['k'],
    # ]
    #
    # wf = wf.derive.split(
    #     '0-0', 'f0',
    #     [a3 @ td.ts['k'], a3 @ td.ts['k-1']],
    #     ['+', '-'],
    #     factors=[1/dt, 1/dt],
    # )
    #
    # wf = wf.derive.split(
    #     '0-2', 'f0',
    #     [ph.d(b2 @ td.ts['k-1']), ph.d(b2 @ td.ts['k'])],
    #     ['+', '+'],
    #     factors=[1/2, 1/2],
    # )
    #
    # wf = wf.derive.split(
    #     '1-0', 'f0',
    #     [b2 @ td.ts['k'], b2 @ td.ts['k-1']],
    #     ['+', '-'],
    #     factors=[1/dt, 1/dt]
    # )
    #
    # wf = wf.derive.split(
    #     '1-2', 'f0',
    #     [a3 @ td.ts['k-1'], a3 @ td.ts['k']],
    #     ['+', '+'],
    #     factors=[1/2, 1/2],
    # )
    #
    # wf = wf.derive.rearrange(
    #     {
    #         0: '0, 3 = 1, 2',
    #         1: '3, 0 = 2, 1, 4',
    #     }
    # )
    #
    # ph.space.finite(3)
    #
    # # ph.list_spaces()
    #
    # # (a3 @ td.ts['k']).ap(r"\vec{\alpha}")
    #
    # # wf.pr()
    #
    # ap = wf.ap()
    # ap.pr()
    #
    #
    #
    # # ap.pr()
    # # print(wf.unknowns, wf.test_forms)
    # # print(ap.test_vectors)
